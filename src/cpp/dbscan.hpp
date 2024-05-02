#ifndef DBSCAN_H
#define DBSCAN_H

#include <cstdbool>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>

namespace density {

    template <typename T>
    class DBSCAN {

    private:
        T m_epsilon;
        long int m_min_points;
        T (* m_distance)(std::vector<T>, std::vector<T>);

        void expand_cluster(const std::vector<std::vector<T> > &distance_matrix, const int index, std::vector<int> index_neighbors, std::vector<int> &clusters, int cluster_id) {
            std::vector<int> seed_neighbors = index_neighbors, n_neighbors, visited;
            visited.push_back(index);
            std::vector<T> neighbor;
            int n_index, n_index_cluster, seed;
            while (!seed_neighbors.empty()) {
                seed = seed_neighbors.back();
                seed_neighbors.pop_back();
                if(std::find(visited.begin(), visited.end(), seed) != visited.end()) {
                    continue;
                }
                visited.push_back(seed);
                n_neighbors = neighbors(distance_matrix, seed);
                if (static_cast<long int>(n_neighbors.size()) < m_min_points) {
                    continue;
                }
                for(auto n_it = n_neighbors.begin(); n_it != n_neighbors.end(); ++n_it) {
                    n_index = *n_it;
                    n_index_cluster = clusters.at(n_index);
                    if (n_index_cluster == -1 || n_index_cluster == -2) {
                        if (n_index_cluster == -2) {
                            seed_neighbors.push_back(n_index);
                        }
                        clusters.at(n_index) = cluster_id;
                    }
                }
            }
        }

        std::vector<int> neighbors(const std::vector<std::vector<T> > &distance_matrix, size_t index) {
            T distance;
            std::vector<int> output;
            std::vector<T> row = distance_matrix.at(index);
            for(auto it = row.begin(); it != row.end(); ++it) {
                int index = std::distance(row.begin(), it);
                distance = row.at(index);
                if (distance < m_epsilon) {
                    output.push_back(index);
                }
            }
            return output;
        }

        virtual std::vector<std::vector<T> > calculate_distances(const std::vector<std::vector<T> > &data) {
            size_t data_size = data.size(), i = 0, j = 0;
            std::vector<std::vector<T> > output(data_size);
            for(i = 0; i < data_size; ++i) {
                std::vector<T> row(data_size);
                output.at(i) = row;
            }
            #pragma omp parallel for if(data_size > 2000) private(i, j) shared(data)
            for (i = 0; i < data_size; ++i) {
                std::vector<T> point1 = data.at(i), point2;
                for(j = i; j < data_size; ++j) {
                    T dist = m_distance(point1, data.at(j));
                    #pragma omp critical
                    output[i][j] = dist;
                    #pragma omp critical
                    output[j][i] = dist;
                }
            }
            return output;
        }

    public:
        DBSCAN(){};
        DBSCAN(const T epsilon, const long int min_points, T (* distance_func)(std::vector<T>, std::vector<T>)) {
            assert(epsilon > 0);
            assert(min_points > 0);
            m_epsilon = epsilon;
            m_min_points = min_points;
            m_distance = distance_func;
        }
        virtual ~DBSCAN() {};

        void setEpsilon(const long int epsilon) {
            this->m_epsilon = epsilon;
        }

        long int getEpsilon() {
            return this->m_epsilon;
        }

        void setMinPoints (const long int minPoints) {
            this->m_min_points = minPoints;
        }

        long int getMinPoints () {
            return this->m_min_points;
        }

        std::vector<int> predict(std::vector<std::vector<T> > data) {
            const std::size_t sample_count = data.size();
            std::vector<int> clusters(sample_count);

            for(auto it = clusters.begin(); it != clusters.end(); ++it) {
                *it = -2;
            }

            const std::vector<std::vector<T> > distance_matrix = this->calculate_distances(data);
            int cluster_id = 0;
            size_t index;
            for(auto it = data.begin(); it != data.end(); ++it) {
                index = std::distance(data.begin(), it);
                if (clusters.at(index) != -2) {
                    continue;
                }
                std::vector<int> point_neighbors = neighbors(distance_matrix, index);
                if (static_cast<long int>(point_neighbors.size()) < m_min_points) {
                    clusters.at(index) = -1;
                    continue;
                }
                clusters.at(index) = cluster_id;
                expand_cluster(distance_matrix, index, point_neighbors, clusters, cluster_id);
                cluster_id += 1;
            }
            return clusters;
        }
    };

    template <typename T>
    class DBPack {
        // A special case of DBSCAN, where points have one dimension and are
        // sorted.  Can be used for clustering events based on when they occurred
        // can be used for online clustering where new points are sorted relative
        // to existing points

    private:
        T m_epsilon;
        unsigned long int m_min_points;

        std::vector<size_t> neighbors(const std::vector<T> &data, size_t &max_index) {
            size_t sample_count = data.size();
            T current_point = data.at(max_index), neighbor;
            T distance = 0.0;
            size_t neighbor_index = max_index;
            std::vector<size_t> output;
            output.push_back(max_index);
            while (distance < m_epsilon && neighbor_index > 0) {
                --neighbor_index;
                neighbor = data.at(neighbor_index);
                distance = abs(neighbor - current_point);
                if (distance > m_epsilon) {
                    break;
                }
                output.push_back(neighbor_index);
            }
            distance = 0.0;
            neighbor_index = max_index;
            while (distance < m_epsilon && neighbor_index < sample_count - 1) {
                ++neighbor_index;
                neighbor = data.at(neighbor_index);
                distance = abs(neighbor - current_point);
                if (distance > m_epsilon) {
                    break;
                }
                output.push_back(neighbor_index);
            }
            return output;
        }

        void expand_cluster(const std::vector<T> &data, size_t &max_index, std::vector<size_t> index_neighbors, std::vector<int> &clusters, int cluster_id) {
            std::vector<size_t> seed_neighbors = index_neighbors, n_neighbors;
            std::vector<T> neighbor;
            size_t sample_count = data.size();
            while (max_index < sample_count) {
                n_neighbors = neighbors(data, max_index);
                if (static_cast<unsigned long int>(n_neighbors.size()) < m_min_points) {
                    clusters.at(max_index) = -1;
                    break;
                }
                clusters.at(max_index) = cluster_id;
                ++max_index;
            }
        }

    public:
        DBPack(const T epsilon, const unsigned long int min_points) {
            assert(epsilon > 0);
            assert(min_points > 0);
            m_epsilon = epsilon;
            m_min_points = min_points;
        }
        ~DBPack() {};

        void setEpsilon(const long int epsilon) {
            this->m_epsilon = epsilon;
        }

        long int getEpsilon() {
            return this->m_epsilon;
        }

        void setMinPoints (const long int minPoints) {
            this->m_min_points = minPoints;
        }

        long int getMinPoints () {
            return this->m_min_points;
        }

        std::vector<int> predict(const std::vector<T> &data) {
            const std::size_t sample_count = data.size();
            std::vector<int> clusters(sample_count);
            size_t max_index = 0;
            int cluster_id = 0;
            while (max_index < sample_count) {
                std::vector<size_t> point_neighbors = this->neighbors(data, max_index);
                if (static_cast<unsigned long int>(point_neighbors.size()) < m_min_points) {
                    clusters.at(max_index) = -1;
                    ++max_index;
                    continue;
                }
                clusters.at(max_index) = cluster_id;
                expand_cluster(data, max_index, point_neighbors, clusters, cluster_id);
                ++cluster_id;
                ++max_index;
            }
            return clusters;


        }
    };

};

#endif /* DBSCAN_H */
