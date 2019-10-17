#include <cstdbool>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include "distance.cpp"

namespace density {

    namespace fuzzy {

        void print_vector(std::vector<size_t> data) {
            for (auto i = data.begin(); i != data.end(); ++i) {
                std::cout << *i << ", ";
            }
            std::cout << "\n";
        }

        template <typename T>
        class BaseDBSCAN {

        protected:
            unsigned long int m_min_points;
            long double (* m_distance)(std::vector<T>, std::vector<T>);

            virtual std::vector<size_t> neighbors(std::vector<std::vector<T> > &data, std::vector<T> point, long double epsilon) {
                long double distance = 0.0;
                std::vector<size_t> output;
                for(auto it = data.begin(); it != data.end(); ++it) {
                    size_t index = std::distance(data.begin(), it);
                    distance = m_distance(*it, point);
                    if (distance < epsilon) {
                        output.push_back(index);
                    }
                }
                return output;
            }

        public:
            BaseDBSCAN(const unsigned long int min_points, long double (* distance_func)(std::vector<T>, std::vector<T>)) {
                assert(min_points > 0);
                m_min_points = min_points;
                m_distance = distance_func;
            }
            std::vector<std::unordered_map<int, long double> > predict(std::vector<std::vector<T> > &data);
        };

        template <typename T>
        class CoreDBSCAN: public BaseDBSCAN<T> {

        private:
            long double m_epsilon;
            unsigned long int m_min_points;
            unsigned long int m_max_points;
            long double (* m_distance)(std::vector<T>, std::vector<T>);

            void expand_cluster(std::vector<std::vector<T> > &data, const size_t index, std::vector<size_t> index_neighbors, std::vector<std::unordered_map<int, long double> > &clusters, int cluster_id) {
                std::vector<size_t> seed_neighbors = index_neighbors, n_neighbors, n_n_neighbors, visited;
                visited.push_back(index);
                std::vector<T> neighbor;
                int n_index, seed;
                long double cluster_membership, min_membership;
                bool assigned;
                std::vector<T> n_point;
                while (!seed_neighbors.empty()) {
                    seed = seed_neighbors.front();
                    seed_neighbors.erase(seed_neighbors.begin());
                    if(std::find(visited.begin(), visited.end(), seed) != visited.end()) {
                        continue;
                    }
                    visited.push_back(seed);
                    neighbor = data.at(seed);
                    n_neighbors = neighbors(data, neighbor,m_epsilon);
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
                            n_point = data.at(n_index);
                            n_n_neighbors = neighbors(data, n_point, m_epsilon);
                            cluster_membership = membership(n_n_neighbors);
                            if (cluster_membership > 0 && cluster_membership < min_membership) {
                                min_membership = cluster_membership;
                            }
                        }
                        clusters.at(seed)[cluster_id] = min_membership;
                    }
                }
            }

            long double membership(std::vector<size_t> neighborhood) {
                size_t neighborhood_size = neighborhood.size();
                if (neighborhood_size >= m_max_points) {
                    return 1.0;
                } else if (neighborhood_size <= m_min_points) {
                    return 0.0;
                }
                long double min_max_difference = m_max_points - m_min_points;
                long double neighbor_difference = (long double) neighborhood_size - m_min_points;
                return neighbor_difference / min_max_difference;
            }

            public:
                CoreDBSCAN(const long double epsilon, const unsigned long int min_points, const unsigned long int max_points, long double (* distance_func)(std::vector<T>, std::vector<T>)): BaseDBSCAN<T>(min_points, distance_func) {
                    assert(epsilon > 0);
                    assert(min_points > 0);
                    assert(max_points >= min_points);
                    m_epsilon = epsilon;
                    m_min_points = min_points;
                    m_max_points = max_points;
                    m_distance = distance_func;
                }

                std::vector<std::unordered_map<int, long double> > predict(std::vector<std::vector<T> > &data) {
                    const std::size_t sample_count = data.size();
                    std::vector<std::unordered_map<int, long double> > clusters(sample_count);

                    int cluster_id = 0;
                    int index;
                    long double cluster_membership;
                    for(auto it = data.begin(); it != data.end(); ++it) {
                        index = std::distance(data.begin(), it);
                        if (!clusters.at(index).empty()) {
                            continue;
                        }

                        std::vector<size_t> point_neighbors = neighbors(data, *it, m_epsilon);
                        if (point_neighbors.size() < m_min_points) {
                            clusters.at(index)[-1] = 1.0;
                        }
                        else {
                            cluster_membership = membership(point_neighbors);
                            clusters.at(index)[cluster_id] = cluster_membership;
                            expand_cluster(data, index, point_neighbors, clusters, cluster_id);
                            cluster_id += 1;
                         }
                    }
                    return clusters;
                }
        };

        template <typename T>
        class BorderDBSCAN: public BaseDBSCAN<T> {

        private:
            long double m_min_epsilon;
            long double m_max_epsilon;
            unsigned long int m_min_points;
            long double (* m_distance)(std::vector<T>, std::vector<T>);

            void expand_cluster(std::vector<std::vector<T> > &data, const size_t index, std::vector<size_t> index_neighbors, std::vector<std::unordered_map<int, long double> > &clusters, int cluster_id, std::vector<bool> &visited) {
                std::vector<size_t> n_neighbors, fuzzy_border_points, n_fuzzy_border_points;
                clusters.at(index)[cluster_id] = 1.0;
                std::vector<size_t> core = {index};
                std::vector<T> index_point = data.at(index);
                fuzzy_border_points = this->neighbors(data, index_point, m_max_epsilon);
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
                    neighbor = data.at(seed);
                    n_neighbors = this->neighbors(data, neighbor, m_min_epsilon);
                    if (n_neighbors.size() > m_min_points) {
                        n_fuzzy_border_points = this->neighbors(data, neighbor, m_max_epsilon);
                        for(auto it = n_neighbors.begin(); it != n_neighbors.end(); ++it) {
                            if(std::find(index_neighbors.begin(), index_neighbors.end(), *it) == index_neighbors.end()) {
                                index_neighbors.push_back(*it);
                            }
                            // remove min_eps neighborhood from max_eps neighborhood
                            n_fuzzy_border_points.erase(std::remove(n_fuzzy_border_points.begin(), n_fuzzy_border_points.end(), *it), n_fuzzy_border_points.end());
                        }
                        // union all fuzzy border points
                        for(auto it = n_fuzzy_border_points.begin(); it != n_fuzzy_border_points.end(); ++it) {
                            if(std::find(fuzzy_border_points.begin(), fuzzy_border_points.end(), *it) == fuzzy_border_points.end()) {
                                fuzzy_border_points.push_back(*it);
                            };
                        }
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
                long double min_membership;
                long double cluster_membership;
                std::vector<T> fuzzy_border_point;
                long double distance;
                size_t n_seed;

                for(auto it = fuzzy_border_points.begin(); it != fuzzy_border_points.end(); ++it) {
                    fuzzy_border_point = data.at(*it);
                    n_neighbors = this->neighbors(data, fuzzy_border_point, m_max_epsilon);

                    // getting neighbors that are also core objects
                    for(auto n_it = n_neighbors.begin(); n_it != n_neighbors.end();) {
                        if(std::find(core.begin(), core.end(), *n_it) == core.end()) {
                            n_neighbors.erase(std::remove(n_neighbors.begin(), n_neighbors.end(), *n_it), n_neighbors.end());
                        } else {
                            ++n_it;
                        }
                    }
                    min_membership = 1.0;
                    for(auto n_it = n_neighbors.begin(); n_it != n_neighbors.end(); ++n_it) {
                        n_seed = *n_it;
                        n_point = data.at(n_seed);
                        distance = m_distance(fuzzy_border_point, n_point);
                        cluster_membership = membership(distance);
                        if (cluster_membership > 0 && cluster_membership < min_membership) {
                            min_membership = cluster_membership;
                        }
                    }
                    clusters.at(*it)[cluster_id] = min_membership;
                }
            }

            long double membership(long double distance) {
                if (distance <= m_min_epsilon) {
                    return 1.0;
                } else if (distance > m_max_epsilon) {
                    return 0.0;
                }
                long double min_max_difference = m_max_epsilon - m_min_epsilon;
                long double neighbor_difference = m_max_epsilon - distance;
                return neighbor_difference / min_max_difference;
            }

            public:
                BorderDBSCAN(const long double min_epsilon, long double max_epsilon, const unsigned long int min_points, long double (* distance_func)(std::vector<T>, std::vector<T>)): BaseDBSCAN<T>(min_points, distance_func) {
                    assert(min_epsilon > 0);
                    assert(max_epsilon >= min_epsilon);
                    assert(min_points > 0);
                    m_min_epsilon = min_epsilon;
                    m_max_epsilon = max_epsilon;
                    m_min_points = min_points;
                    m_distance = distance_func;
                }

                std::vector<std::unordered_map<int, long double> > predict(std::vector<std::vector<T> > &data) {
                    const std::size_t sample_count = data.size();
                    std::vector<std::unordered_map<int, long double> > clusters(sample_count);
                    std::vector<bool> visited(sample_count);
                    std::vector<int> point_types(sample_count);

                    int index;
                    for(auto it = data.begin(); it != data.end(); ++it) {
                        index = std::distance(data.begin(), it);
                        visited[index] = false;
                    }

                    int cluster_id = 0;
                    for(auto it = data.begin(); it != data.end(); ++it) {
                        index = std::distance(data.begin(), it);
                        if (visited.at(index)) {
                            continue;
                        }

                        visited.at(index) = true;
                        std::vector<size_t> point_neighbors = this->neighbors(data, *it, m_min_epsilon);
                        if (point_neighbors.size() <= m_min_points) {
                            clusters.at(index)[-1] = 1.0;
                        } else {
                            expand_cluster(data, index, point_neighbors, clusters, cluster_id, visited);
                            cluster_id += 1;
                         }
                    }
                    return clusters;
                }
        };

        template <typename T>
        class DBSCAN: BaseDBSCAN<T> {

        private:
            long double m_min_epsilon;
            long double m_max_epsilon;
            unsigned long int m_min_points;
            unsigned long int m_max_points;

            void expand_cluster(std::vector<std::vector<T> > &data, const size_t index, std::vector<size_t> index_neighbors, std::vector<std::unordered_map<int, long double> > &clusters, int cluster_id, std::vector<bool> &visited) {
                std::vector<size_t> n_neighbors, n_n_neighbors, core = {index};
                visited.push_back(index);
                std::vector<T> neighbor;
                long double n_density, n_distance_membership, n_core_membership;
                std::vector<T> n_point;
                size_t seed;
                typename std::vector<T>::size_type i;
                for(i = 0; i < index_neighbors.size(); ++i) {
                    seed = index_neighbors.at(i);
                    visited.at(seed) = true;
                    n_point = data.at(seed);
                    n_neighbors = this->neighbors(data, n_point, m_max_epsilon);
                    n_density = density(data, seed, n_neighbors);
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
                        long double min_membership = 1.0;
                        for (auto n_it = n_neighbors.begin(); n_it != n_neighbors.end(); ++n_it) {
                            n_distance_membership = distance_membership(m_distance(n_point, data.at(*n_it)));
                            n_n_neighbors = this->neighbors(data, data.at(*n_it), m_max_epsilon);
                            n_density = density(data, *n_it, n_n_neighbors);
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

            long double core_membership(long double density) {
                if (density >= m_max_points) {
                    return 1.0;
                } else if (density <= m_min_points) {
                    return 0.0;
                }
                long double min_max_difference = m_max_points - m_min_points;
                long double neighbor_difference = density - m_min_points;
                return neighbor_difference / min_max_difference;
            }

            long double distance_membership(long double distance) {
                if (distance <= m_min_epsilon) {
                    return 1.0;
                } else if (distance > m_max_epsilon) {
                    return 0.0;
                }
                long double min_max_difference = m_max_epsilon - m_min_epsilon;
                long double neighbor_difference = m_max_epsilon - distance;
                return neighbor_difference / min_max_difference;
            }

            long double density(std::vector<std::vector<T> > &data, int index, std::vector<size_t> &neighbors) {
                long double output = 0.0;
                std::vector<T> point = data.at(index);
                for(auto it = neighbors.begin(); it != neighbors.end(); ++it) {
                    if (*it == index) {
                        continue;
                    }
                    output += distance_membership(m_distance(point, data.at(*it)));
                }
                return output;
            }

        protected:
            long double (* m_distance)(std::vector<T>, std::vector<T>);

        public:
            DBSCAN(const long double min_epsilon, long double max_epsilon, const unsigned long int min_points, const unsigned long int max_points, long double (* distance_func)(std::vector<T>, std::vector<T>)): BaseDBSCAN<T>(min_points, distance_func) {
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

            std::vector<std::unordered_map<int, long double> > predict(std::vector<std::vector<T> > &data) {
                const std::size_t sample_count = data.size();
                std::vector<std::unordered_map<int, long double> > clusters(sample_count);
                std::vector<bool> visited(sample_count);
                std::vector<int> point_types(sample_count);

                int index;
                for(auto it = data.begin(); it != data.end(); ++it) {
                    index = std::distance(data.begin(), it);
                    visited[index] = false;
                }

                int cluster_id = 0;
                long double index_density;
                long double index_core_membership;
                for(auto it = data.begin(); it != data.end(); ++it) {
                    index = std::distance(data.begin(), it);
                    if (visited.at(index)) {
                        continue;
                    }

                    visited.at(index) = true;
                    std::vector<size_t> point_neighbors = this->neighbors(data, *it, m_max_epsilon);
                    index_density = density(data, index, point_neighbors);
                    index_core_membership = core_membership(index_density);
                    if (index_core_membership == 0) {
                        clusters.at(index)[-1] = 1.0;
                    } else {
                        cluster_id += 1;
                        clusters.at(index)[cluster_id] = index_core_membership;
                        expand_cluster(data, index, point_neighbors, clusters, cluster_id, visited);
                     }
                }
                return clusters;
            }
        };
    }
}

void print_map(std::unordered_map<int, long double> data) {
    std::cout << "{";
    for (auto x : data) {
        std::cout << x.first << ": " << x.second << ", ";
    }
    std::cout << "}";
}

int main() {
    std::vector<std::vector<long double> > data = {{931.0}, {931.0}, {932.0}, {932.0}, {932.0}, {932.0}, {932.0}, {932.0}, {933.0}, {933.0}, {933.0}, {933.0}, {933.0}, {933.0}, {933.0}, {933.0}, {933.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {935.0}, {935.0}, {935.0}, {935.0}, {935.0}, {936.0}, {936.0}, {936.0}, {936.0}, {936.0}, {936.0}, {937.0}, {938.0}, {938.0}, {938.0}, {938.0}, {938.0}, {939.0}, {939.0}, {939.0}, {939.0}, {939.0}, {940.0}, {940.0}, {940.0}, {940.0}, {941.0}, {941.0}, {941.0}, {942.0}, {942.0}, {942.0}, {943.0}, {944.0}, {944.0}, {945.0}, {945.0}, {945.0}, {945.0}, {946.0}, {946.0}, {947.0}, {947.0}, {947.0}, {948.0}, {948.0}, {948.0}, {949.0}, {949.0}, {949.0}, {949.0}, {949.0}, {950.0}, {950.0}, {950.0}, {950.0}, {951.0}, {951.0}, {952.0}, {953.0}, {953.0}, {955.0}, {955.0}, {965.0}, {966.0}, {966.0}, {966.0}, {966.0}, {967.0}, {968.0}, {968.0}, {968.0}, {968.0}, {969.0}, {969.0}, {970.0}, {970.0}, {970.0}, {971.0}, {971.0}, {972.0}, {972.0}, {972.0}, {973.0}, {973.0}, {974.0}, {980.0}, {980.0}, {981.0}, {981.0}, {981.0}, {982.0}, {983.0}, {983.0}, {983.0}, {983.0}, {984.0}, {984.0}, {994.0}, {994.0}, {996.0}, {1002.0}, {1007.0}, {1007.0}, {1007.0}, {1007.0}, {1008.0}, {1009.0}, {1009.0}, {1010.0}, {1028.0}, {1030.0}, {1061.0}, {1078.0}};
    std::vector<std::vector<long double> > other_data = {{7344.0}, {7380.0}, {7392.0}, {7451.0}, {7466.0}, {7478.0}, {7493.0}, {7499.0}, {7499.0}, {7510.0}, {7543.0}, {7563.0}, {7569.0}, {7569.0}, {7580.0}, {7591.0}, {7609.0}, {7620.0}, {7623.0}, {7631.0}, {7638.0}, {7645.0}, {7663.0}, {7665.0}, {7667.0}, {7686.0}, {7691.0}, {7701.0}, {7701.0}, {7702.0}, {7735.0}, {7750.0}, {7755.0}, {7760.0}, {7777.0}, {7790.0}, {7796.0}, {7797.0}, {7805.0}, {7809.0}, {7811.0}, {7814.0}, {7819.0}, {7820.0}, {7821.0}, {7828.0}, {7833.0}, {7849.0}, {7853.0}, {7853.0}, {7862.0}, {7874.0}, {7877.0}, {7878.0}, {7880.0}, {7886.0}, {7891.0}, {7894.0}, {7896.0}, {7897.0}, {7899.0}, {7900.0}, {7904.0}, {7929.0}, {7945.0}, {7953.0}, {7958.0}, {7961.0}, {7963.0}, {7964.0}, {7970.0}, {7978.0}, {7998.0}, {7998.0}, {7999.0}, {8021.0}, {8021.0}, {8025.0}, {8033.0}, {8056.0}, {8062.0}, {8063.0}, {8070.0}, {8074.0}, {8110.0}, {8113.0}, {8118.0}, {8119.0}, {8125.0}, {8137.0}, {8151.0}, {8151.0}, {8152.0}, {8169.0}, {8192.0}, {8214.0}, {8237.0}, {8249.0}, {8268.0}, {8275.0}, {8278.0}, {8284.0}, {8285.0}, {8303.0}, {8304.0}, {8308.0}, {8322.0}, {8345.0}, {8352.0}, {8361.0}, {8365.0}, {8370.0}, {8380.0}, {8383.0}, {8394.0}, {8416.0}, {8445.0}, {8454.0}, {8457.0}, {8490.0}, {8506.0}, {8512.0}, {8520.0}, {8533.0}, {8540.0}, {8545.0}, {8563.0}, {8569.0}, {8590.0}, {8611.0}, {8810.0}, {8834.0}, {8850.0}, {8858.0}, {8882.0}, {8895.0}, {8896.0}, {8904.0}, {9148.0}, {9347.0}, {9419.0}};
    //long double epsilon = 5;
    //unsigned long int min_points = 2;
    //unsigned long int max_points = 4;
    //density::fuzzy::CoreDBSCAN<long double> clf = density::fuzzy::CoreDBSCAN<long double>(epsilon, min_points, max_points, euclidean);
    //std::vector<std::unordered_map<int, long double> > clusters = clf.predict(data);
    /*long double min_epsilon = 2.1;
    long double max_epsilon = 6;
    unsigned long int min_points = 1;
    density::fuzzy::BorderDBSCAN<long double> clf = density::fuzzy::BorderDBSCAN<long double>(min_epsilon, max_epsilon, min_points, euclidean);
    std::vector<std::unordered_map<int, long double> > clusters = clf.predict(other_data);
    for (auto i = clusters.begin(); i != clusters.end(); ++i) {
        size_t index = std::distance(clusters.begin(), i);
        std::cout << "Index: " << index << ", Point: " << other_data.at(index).at(0) << "    ";
        print_map(*i);
        std::cout << '\n';
    }*/

    long double min_epsilon = 1;
    long double max_epsilon = 8;
    unsigned long int min_points = 1;
    unsigned long int max_points = 3;
    density::fuzzy::DBSCAN<long double> clf = density::fuzzy::DBSCAN<long double>(min_epsilon, max_epsilon, min_points, max_points, euclidean);
    std::vector<std::unordered_map<int, long double> > clusters = clf.predict(other_data);
    for (auto i = clusters.begin(); i != clusters.end(); ++i) {
        size_t index = std::distance(clusters.begin(), i);
        std::cout << "Index: " << index << ", Point: " << other_data.at(index).at(0) << "    ";
        print_map(*i);
        std::cout << '\n';
    }
}
