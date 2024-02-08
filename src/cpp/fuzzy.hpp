#ifndef FUZZY_H
#define FUZZY_H

#include <cstdbool>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>


inline std::vector<size_t> vector_intersection(std::vector<size_t> &v1, std::vector<size_t> &v2){
    std::vector<std::size_t> v3;

    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());

    std::set_intersection(v1.begin(),v1.end(), v2.begin(),v2.end(), std::back_inserter(v3));
    return v3;
}

inline std::vector<size_t> vector_union(std::vector<size_t> &v1, std::vector<size_t> &v2){
    std::vector<std::size_t> v3;

    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());

    std::set_union(v1.begin(),v1.end(), v2.begin(),v2.end(), std::back_inserter(v3));
    return v3;
}

namespace density {

    namespace fuzzy {


        template <typename T>
        class BaseDBSCAN {

        protected:
            unsigned long int m_min_points;
            double (* m_distance)(std::vector<T>, std::vector<T>);

            virtual std::vector<size_t> neighbors(const std::vector<std::vector<T> > &distance_matrix, const size_t index, const double &epsilon) {
                std::vector<size_t> output;
                std::vector<T> row = distance_matrix.at(index);
                for(auto it = row.begin(); it != row.end(); ++it) {
                    size_t index = std::distance(row.begin(), it);
                    if (*it < epsilon) {
                        output.push_back(index);
                    }
                }
                return output;
            }

            virtual std::vector<std::vector<double> > calculate_distances(const std::vector<std::vector<T> > &data) {
                size_t data_size = data.size(), i = 0, j = 0;
                std::vector<std::vector<double> > output(data_size);


                for(i = 0; i < data_size; ++i) {
                    std::vector<double> row(data_size);
                    output.at(i) = row;
                }
                #pragma omp parallel for if(data_size > 2000) private(i, j) shared(data)
                for (i = 0; i < data_size; ++i) {
                    std::vector<T> point1, point2;
                    point1 = data.at(i);
                    for(j = i; j < data_size; ++j) {
                        point2 = data.at(j);
                        double dist;
                        if (j == i || point1 == point2) {
                            dist = 0;
                        } else {
                            dist = m_distance(point1, point2);
                        }
                        #pragma omp critical
                        output[i][j] = dist;
                        #pragma omp critical
                        output[j][i] = dist;
                    }
                }
                return output;
            }

        public:
            BaseDBSCAN(const unsigned long int min_points, double (* distance_func)(std::vector<T>, std::vector<T>)) {
                assert(min_points > 0);
                m_min_points = min_points;
                m_distance = distance_func;
            }
            virtual ~BaseDBSCAN() {};
            std::vector<std::map<int, double> > predict(std::vector<std::vector<T> > &data) {
                std::vector<std::map<int, double> > clusters;
                return clusters;
            };
        };

        template <typename T>
        class CoreDBSCAN: public BaseDBSCAN<T> {

        private:
            double m_epsilon;
            unsigned long int m_min_points;
            unsigned long int m_max_points;
            double (* m_distance)(std::vector<T>, std::vector<T>);

            void expand_cluster(const std::vector<std::vector<T> > &distance_matrix, const size_t index, std::vector<size_t> index_neighbors, std::vector<std::map<int, double> > &clusters, int cluster_id) {
                std::vector<size_t> seed_neighbors = index_neighbors, n_neighbors, n_n_neighbors, visited;
                visited.push_back(index);
                std::vector<T> neighbor;
                int n_index, seed;
                double cluster_membership, min_membership;
                bool assigned;
                std::vector<T> n_point;
                while (!seed_neighbors.empty()) {
                    seed = seed_neighbors.front();
                    seed_neighbors.erase(seed_neighbors.begin());
                    if(std::find(visited.begin(), visited.end(), seed) != visited.end()) {
                        continue;
                    }
                    visited.push_back(seed);
                    n_neighbors = this->neighbors(distance_matrix, seed, m_epsilon);
                    if (n_neighbors.size() >= m_min_points) {
                        for(auto n_it = n_neighbors.begin(); n_it != n_neighbors.end(); ++n_it) {
                            n_index = *n_it;
                            if(std::find(visited.begin(), visited.end(), n_index) != visited.end()) {
                                continue;
                            }
                            seed_neighbors.push_back(n_index);
                        }
                        clusters.at(seed)[cluster_id] = membership(n_neighbors);
                    }
                    assigned = clusters.at(seed).empty() || clusters.at(seed).find(-1) != clusters.at(seed).end();
                    if (assigned) {
                        min_membership = 1.0;
                        for(auto n_it = n_neighbors.begin(); n_it != n_neighbors.end(); ++n_it) {
                            n_index = *n_it;
                            n_n_neighbors = this->neighbors(distance_matrix, n_index, m_epsilon);
                            cluster_membership = membership(n_n_neighbors);
                            if (cluster_membership > 0 && cluster_membership < min_membership) {
                                min_membership = cluster_membership;
                            }
                        }
                        clusters.at(seed)[cluster_id] = min_membership;
                    }
                }
            }

            double membership(const std::vector<size_t> &neighborhood) {
                size_t neighborhood_size = neighborhood.size();
                if (neighborhood_size >= m_max_points) {
                    return 1.0;
                } else if (neighborhood_size <= m_min_points) {
                    return 0.0;
                }
                double min_max_difference = m_max_points - m_min_points;
                double neighbor_difference = (double) neighborhood_size - m_min_points;
                return neighbor_difference / min_max_difference;
            }

            public:
                CoreDBSCAN(const double epsilon, const unsigned long int min_points, const unsigned long int max_points, double (* distance_func)(std::vector<T>, std::vector<T>)): BaseDBSCAN<T>(min_points, distance_func) {
                    assert(epsilon > 0);
                    assert(min_points > 0);
                    assert(max_points >= min_points);
                    m_epsilon = epsilon;
                    m_min_points = min_points;
                    m_max_points = max_points;
                    m_distance = distance_func;
                }
                ~CoreDBSCAN() {};

                std::vector<std::map<int, double> > predict(const std::vector<std::vector<T> > &data) {
                    const std::size_t sample_count = data.size();
                    std::vector<std::map<int, double> > clusters(sample_count);

                    int cluster_id = 0;
                    int index;
                    double cluster_membership;

                    const std::vector<std::vector<double> > distance_matrix = this->calculate_distances(data);
                    for(auto it = data.begin(); it != data.end(); ++it) {
                        index = std::distance(data.begin(), it);
                        if (!clusters.at(index).empty()) {
                            continue;
                        }

                        std::vector<size_t> point_neighbors = this->neighbors(distance_matrix, index, m_epsilon);
                        if (point_neighbors.size() < m_min_points) {
                            clusters.at(index)[-1] = 1.0;
                        }
                        else {
                            cluster_membership = membership(point_neighbors);
                            clusters.at(index)[cluster_id] = cluster_membership;
                            expand_cluster(distance_matrix, index, point_neighbors, clusters, cluster_id);
                            cluster_id += 1;
                         }
                    }
                    return clusters;
                }
        };

        template <typename T>
        class BorderDBSCAN: public BaseDBSCAN<T> {

        private:
            double m_min_epsilon;
            double m_max_epsilon;
            unsigned long int m_min_points;
            double (* m_distance)(std::vector<T>, std::vector<T>);

            void expand_cluster(const std::vector<std::vector<T> > &distance_matrix, const size_t index, std::vector<size_t> index_neighbors, std::vector<std::map<int, double> > &clusters, int cluster_id, std::vector<bool> &visited) {
                std::vector<size_t> n_neighbors, fuzzy_border_points, n_fuzzy_border_points;
                clusters.at(index)[cluster_id] = 1.0;
                std::vector<size_t> core = {index};
                fuzzy_border_points = this->neighbors(distance_matrix, index, m_max_epsilon);
                // remove core points from  fuzzy border points
                for(auto it = index_neighbors.begin(); it != index_neighbors.end(); ++it) {
                    fuzzy_border_points.erase(std::remove(fuzzy_border_points.begin(), fuzzy_border_points.end(), *it), fuzzy_border_points.end());
                }
                std::vector<T> neighbor;
                std::vector<T> n_point;
                size_t seed;
                typename std::vector<T>::size_type i;
                for(i = 0; i < index_neighbors.size(); ++i) {
                    seed = index_neighbors.at(i);
                    visited.at(seed) = true;
                    n_neighbors = this->neighbors(distance_matrix, seed, m_min_epsilon);
                    if (n_neighbors.size() > m_min_points) {
                        n_fuzzy_border_points = this->neighbors(distance_matrix, seed, m_max_epsilon);
                        for(auto it = n_neighbors.begin(); it != n_neighbors.end(); ++it) {
                            if(std::find(index_neighbors.begin(), index_neighbors.end(), *it) == index_neighbors.end()) {
                                index_neighbors.push_back(*it);
                            }
                            // remove min_eps neighborhood from max_eps neighborhood
                            n_fuzzy_border_points.erase(std::remove(n_fuzzy_border_points.begin(), n_fuzzy_border_points.end(), *it), n_fuzzy_border_points.end());
                        }
                        // union all fuzzy border points
                        fuzzy_border_points = vector_union(fuzzy_border_points, n_fuzzy_border_points);
                        // add seed as core point to cluster
                        clusters.at(seed)[cluster_id] = 1.0;
                        core.push_back(seed);
                    } else {
                        fuzzy_border_points.push_back(seed);
                    }
                }
                // remove Core objects from fuzzy points
                for(auto it = core.begin(); it != core.end(); ++it) {
                    if(std::find(fuzzy_border_points.begin(), fuzzy_border_points.end(), *it) != fuzzy_border_points.end()) {
                        fuzzy_border_points.erase(std::remove(fuzzy_border_points.begin(), fuzzy_border_points.end(), *it), fuzzy_border_points.end());
                    };
                }
                // process fuzzy border points
                i = 0;
                size_t fuzzy_border_point_count = fuzzy_border_points.size();
                #pragma omp parallel for if(fuzzy_border_point_count > 500) private(i, n_neighbors)
                for(i = 0; i < fuzzy_border_point_count; ++i) {
                    size_t point_index = fuzzy_border_points[i];
                    n_neighbors = this->neighbors(distance_matrix, point_index, m_max_epsilon);
                    // getting neighbors that are also core objects
                    std::vector<size_t> neighbor_core = vector_intersection(n_neighbors, core);
                    double min_membership = 1.0;
                    for(auto n_it = neighbor_core.begin(); n_it != neighbor_core.end(); ++n_it) {
                        size_t n_seed = *n_it;
                        double distance = distance_matrix.at(point_index).at(n_seed);
                        double cluster_membership = membership(distance);
                        if (cluster_membership > 0 && cluster_membership < min_membership) {
                            min_membership = cluster_membership;
                        }
                    }
                    clusters.at(point_index)[cluster_id] = min_membership;
                }

            }

            double membership(const double distance) {
                if (distance <= m_min_epsilon) {
                    return 1.0;
                } else if (distance > m_max_epsilon) {
                    return 0.0;
                }
                const double min_max_difference = m_max_epsilon - m_min_epsilon;
                const double neighbor_difference = m_max_epsilon - distance;
                return neighbor_difference / min_max_difference;
            }

        public:
            BorderDBSCAN(const double min_epsilon, const double max_epsilon, const unsigned long int min_points, double (* distance_func)(std::vector<T>, std::vector<T>)): BaseDBSCAN<T>(min_points, distance_func) {
                assert(min_epsilon > 0);
                assert(max_epsilon >= min_epsilon);
                assert(min_points > 0);
                m_min_epsilon = min_epsilon;
                m_max_epsilon = max_epsilon;
                m_min_points = min_points;
                m_distance = distance_func;
            }
            ~BorderDBSCAN() {};
            
            std::vector<std::map<int, double> > predict(const std::vector<std::vector<T> > &data) {
                const std::size_t sample_count = data.size();
                std::vector<std::map<int, double> > clusters(sample_count);
                std::vector<bool> visited(sample_count);
                std::vector<int> point_types(sample_count);
                int index;
                for(auto it = data.begin(); it != data.end(); ++it) {
                    index = std::distance(data.begin(), it);
                    visited[index] = false;
                }
                const std::vector<std::vector<double> > distance_matrix = this->calculate_distances(data);
                int cluster_id = 0;
                for(auto it = data.begin(); it != data.end(); ++it) {
                    index = std::distance(data.begin(), it);
                    if (visited.at(index)) {
                        continue;
                    }

                    visited.at(index) = true;
                    std::vector<size_t> point_neighbors = this->neighbors(distance_matrix, index, m_min_epsilon);
                    if (point_neighbors.size() <= m_min_points) {
                        clusters.at(index)[-1] = 1.0;
                    } else {
                        expand_cluster(distance_matrix, index, point_neighbors, clusters, cluster_id, visited);
                        cluster_id += 1;
                     }
                }
                return clusters;
            }
        };

        template <typename T>
        class DBSCAN: public BaseDBSCAN<T> {

        private:
            double m_min_epsilon;
            double m_max_epsilon;
            unsigned long int m_min_points;
            unsigned long int m_max_points;

            void expand_cluster(const std::vector<std::vector<T> > &distance_matrix, const size_t index, std::vector<size_t> index_neighbors, std::vector<std::map<int, double> > &clusters, int cluster_id, std::vector<bool> &visited) {
                std::vector<size_t> n_neighbors, n_n_neighbors, core = {index};
                visited.push_back(index);
                std::vector<T> neighbor;
                double n_density, n_distance_membership, n_core_membership;
                std::vector<T> n_point;
                size_t seed;
                typename std::vector<T>::size_type i;
                for(i = 0; i < index_neighbors.size(); ++i) {
                    seed = index_neighbors.at(i);
                    visited.at(seed) = true;
                    n_neighbors = this->neighbors(distance_matrix, seed, m_max_epsilon);
                    n_density = density(distance_matrix, seed, n_neighbors);
                    n_core_membership = core_membership(n_density);
                    if (n_core_membership > 0) {
                        // core point, adding neighbor to seeds if not already a seed
                        for (auto n_it = n_neighbors.begin(); n_it != n_neighbors.end(); ++n_it) {
                            if(std::find(index_neighbors.begin(), index_neighbors.end(), *n_it) == index_neighbors.end()) {
                                index_neighbors.push_back(*n_it);
                            }
                        }
                        core.push_back(seed);
                        clusters.at(seed)[cluster_id] = n_core_membership;
                    } else {
                        // border point
                        double min_membership = 1.0;
                        for (auto n_it = n_neighbors.begin(); n_it != n_neighbors.end(); ++n_it) {
                            n_distance_membership = distance_membership(distance_matrix.at(seed).at(*n_it));
                            n_n_neighbors = this->neighbors(distance_matrix, *n_it, m_max_epsilon);
                            n_density = density(distance_matrix, *n_it, n_n_neighbors);
                            n_core_membership = core_membership(n_density);
                            if (n_core_membership <= 0 || n_distance_membership <= 0) {
                                continue;
                            }
                            if (n_core_membership < min_membership) {
                                min_membership = n_core_membership;
                            }
                            if (n_distance_membership < min_membership) {
                                min_membership = n_distance_membership;
                            }
                        }
                        clusters.at(seed)[cluster_id] = min_membership;
                    }
                }
            }

            double core_membership(const double density) {
                if (density >= m_max_points) {
                    return 1.0;
                } else if (density <= m_min_points) {
                    return 0.0;
                }
                const double min_max_difference = m_max_points - m_min_points;
                const double neighbor_difference = density - m_min_points;
                return neighbor_difference / min_max_difference;
            }

            double distance_membership(const double distance) {
                if (distance <= m_min_epsilon) {
                    return 1.0;
                } else if (distance > m_max_epsilon) {
                    return 0.0;
                }
                const double min_max_difference = m_max_epsilon - m_min_epsilon;
                const double neighbor_difference = m_max_epsilon - distance;
                return neighbor_difference / min_max_difference;
            }

            double density(const std::vector<std::vector<T> > &distance_matrix, const int index, const std::vector<size_t> &neighbors) {
                double output = 0.0;
                const std::vector<double> row = distance_matrix.at(index);
                for(auto it = neighbors.begin(); it != neighbors.end(); ++it) {
                    output += distance_membership(row.at(*it));
                }
                return output;
            }

        protected:
            double (* m_distance)(std::vector<T>, std::vector<T>);

        public:
            DBSCAN(const double min_epsilon, const double max_epsilon, const unsigned long int min_points, const unsigned long int max_points, double (* distance_func)(std::vector<T>, std::vector<T>)): BaseDBSCAN<T>(min_points, distance_func) {
                assert(min_epsilon > 0);
                assert(max_epsilon >= min_epsilon);
                assert(min_points > 0);
                assert(max_points >= min_points);
                m_min_epsilon = min_epsilon;
                m_max_epsilon = max_epsilon;
                m_min_points = min_points;
                m_max_points = max_points;
                m_distance = distance_func;
            }
            ~DBSCAN() {};
            std::vector<std::map<int, double> > predict(std::vector<std::vector<T> > &data) {
                const std::size_t sample_count = data.size();
                std::vector<std::map<int, double> > clusters(sample_count);
                std::vector<bool> visited(sample_count);
                std::vector<int> point_types(sample_count);

                int index;
                for(auto it = data.begin(); it != data.end(); ++it) {
                    index = std::distance(data.begin(), it);
                    visited[index] = false;
                }

                const std::vector<std::vector<double> > distance_matrix = this->calculate_distances(data);
                int cluster_id = 0;
                double index_density;
                double index_core_membership;
                for(auto it = data.begin(); it != data.end(); ++it) {
                    index = std::distance(data.begin(), it);
                    if (visited.at(index)) {
                        continue;
                    }
                    visited.at(index) = true;
                    std::vector<size_t> point_neighbors = this->neighbors(distance_matrix, index, m_max_epsilon);
                    index_density = density(distance_matrix, index, point_neighbors);
                    index_core_membership = core_membership(index_density);
                    if (index_core_membership == 0) {
                        clusters.at(index)[-1] = 1.0;
                    } else {
                        cluster_id += 1;
                        clusters.at(index)[cluster_id] = index_core_membership;
                        expand_cluster(distance_matrix, index, point_neighbors, clusters, cluster_id, visited);
                     }
                }
                return clusters;
            }
        };
    }
}

#endif /* FUZZY_H */
