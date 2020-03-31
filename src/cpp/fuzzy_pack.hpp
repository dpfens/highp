

#include <cstdbool>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <stdlib.h>

namespace density {

    namespace fuzzy {

        template <class T1, class T2>
        class BaseDBPack {

        private:
            T1 m_min_eps;
            T2 m_min_points;


        protected:
            virtual std::vector<size_t> neighbors(const std::vector<T1> &data, size_t index, const T1 &epsilon) {
                std::vector<size_t> output;
                T1 start_point = data.at(index);
                for(size_t i = index; i > 0; --i) {
                    T1 distance = abs(start_point - data.at(i));
                    if (distance >= epsilon) {
                        break;
                    }
                    output.push_back(i);
                }

                size_t sample_count = data.size();
                for(size_t i = index + 1; i < sample_count; ++i) {
                    T1 distance = abs(start_point - data.at(i));
                    if (distance >= epsilon) {
                        break;
                    }
                    output.push_back(i);
                }
                return output;
            }

        public:BaseDBPack(const T1 min_eps, const T2 min_points) {
                assert(min_eps > 0);
                m_min_eps = min_eps;
                assert(min_points > 0);
                m_min_points = min_points;
            }

            virtual ~BaseDBPack() {};

            std::vector<std::map<T2, T1> > predict(const std::vector<T1> data) {
                const std::size_t sample_count = data.size();
                std::vector<std::map<T2, T1> > clusters(sample_count);
                return clusters;
            }
        };

        template <class T1, class T2>
        class CoreDBPack: public BaseDBPack<T1, T2> {

        private:
            T1 m_min_eps;
            T2 m_min_points;
            T2 m_max_points;

        protected:
            T1 core_membership(size_t neighbor_count) {
                if (neighbor_count >= m_max_points) {
                    return 1.0;
                } else if (neighbor_count <= m_min_points) {
                    return 0.0;
                }
                T1 difference = m_max_points - m_min_points;
                return (neighbor_count - m_min_points) / difference;
            }

            void expand_cluster(const std::vector<T1> &data, size_t &max_index, std::vector<T1> neighbors, std::vector<std::map<T2, T1> > &clusters, const T2 cluster) {
                clusters.at(max_index) = this->core_membership(neighbors.size());
                const size_t sample_count = data.size();
                ++max_index;
                while (max_index < sample_count) {
                    if(std::find(neighbors.begin(), neighbors.end(), max_index) == neighbors.end()) {
                        break;
                    }
                    std::vector<size_t> n_neighbors = this->neighbors(data, max_index);
                    if (n_neighbors.size() > m_min_points) {
                        clusters.at(max_index)[cluster] = this->core_membership(n_neighbors.size());
                    } else {
                        T1 min_membership = 1.0;
                        for (size_t i = 0; i < n_neighbors.size(); ++i) {
                            std::vector<size_t> n_n_neighbors = this->neighbors(data, neighbors[i]);
                            T1 membership = this->core_membership(n_n_neighbors.size());
                            if (membership > 0.0 && membership < min_membership) {
                                min_membership = membership;
                            }
                        }
                        clusters.at(max_index)[cluster] = min_membership;
                    }
                    neighbors = n_neighbors;
                }
            }

        public:
            CoreDBPack(const T1 min_eps, const T2 min_points, const T2 max_points): BaseDBPack<T1, T2>(min_eps, min_points) {
                assert(min_eps > 0);
                assert(min_points > 0);
                assert(max_points >= min_points);
                m_min_eps = min_eps;
                m_min_points = min_points;
                m_max_points = max_points;
            }
            ~CoreDBPack() {};

            std::vector<std::map<T2, T1> > predict(const std::vector<T1> data) {
                const std::size_t sample_count = data.size();
                std::vector<std::map<T2, T1> > clusters(sample_count);
                size_t cluster = 0;
                size_t max_index = 0;
                while (max_index < sample_count) {
                    std::vector<size_t> neighbors = this->neighbors(data, max_index, m_min_eps);
                    if (neighbors.size() <= m_min_points) {
                        clusters.at(max_index)[-1] = 1.0;
                    } else {
                        this->expand_cluster(data, max_index, neighbors, clusters, cluster);
                        ++cluster;
                    }
                    ++max_index;
                }
                return clusters;
            }
        };

        template <class T1, class T2>
        class BorderDBPack: public BaseDBPack<T1, T2> {
        private:
            T1 m_min_eps;
            T1 m_max_eps;
            T2 m_min_points;
        protected:

            T1 border_membership(const T1 distance) {
                if (distance <= m_min_eps) {
                    return 1.0;
                }
                else if (distance > m_max_eps) {
                    return 0.0;
                }
                T1 min_max_difference = m_max_eps - m_min_eps;
                T1 neighbor_difference = m_max_eps - distance;
                return neighbor_difference / min_max_difference;
            }

            void expand_cluster(const std::vector<T1> &data, size_t &max_index, std::vector<size_t> neighbors, std::vector<std::map<T2, T1> > &clusters, const T2 cluster) {
                const size_t sample_count = data.size();
                clusters.at(max_index)[cluster] = 1.0;
                ++max_index;
                for (size_t i = max_index; i < sample_count; ++i) {
                    if(std::find(neighbors.begin(), neighbors.end(), i) == neighbors.end()) {
                        break;
                    }
                    std::vector<size_t> n_neighbors = this->neighbors(data, i, m_min_eps);
                    if (n_neighbors.size() < m_min_points) {
                        break;
                    }
                    neighbors = n_neighbors;
                    max_index = i;
                    clusters.at(i)[cluster] = 1.0;
                }
            }

            void expand_border_forward(const std::vector<T1> &data, size_t core_index, std::vector<std::map<T2, T1> > &clusters, const T2 cluster) {
                const size_t sample_count = data.size();
                const T1 core_point = data.at(core_index);
                for (size_t i = core_index + 1; i < sample_count; ++i) {
                    T1 distance = abs(core_point - data.at(i));
                    if (distance >= m_max_eps) {
                        break;
                    }
                    clusters.at(i)[cluster] = this->border_membership(distance);
                }
            }

            void expand_border_backward(const std::vector<T1> &data, const size_t core_index, std::vector<std::map<T2, T1> > &clusters, const T2 cluster) {
                const T1 core_point = data.at(core_index);
                for (size_t i = core_index - 1; i > 0; i--) {
                    T1 distance = abs(core_point - data.at(i));
                    if (distance >= m_max_eps) {
                        break;
                    }
                    clusters.at(i)[cluster] = this->border_membership(distance);
                }
            }

        public:
            BorderDBPack(const T1 min_eps, const T1 max_eps, const T2 min_points): BaseDBPack<T1, T2>(min_eps, min_points) {
                assert(min_eps > 0);
                assert(max_eps >= min_eps);
                assert(min_points > 0);
                m_min_eps = min_eps;
                m_max_eps = max_eps;
                m_min_points = min_points;
            }
            ~BorderDBPack() {};

            std::vector<std::map<T2, T1> > predict(std::vector<T1> data) {
                const std::size_t sample_count = data.size();
                std::vector<std::map<T2, T1> > clusters(sample_count);
                T2 cluster = 0;
                size_t max_index = 0;
                while (max_index < sample_count) {
                    std::vector<size_t> neighbors = this->neighbors(data, max_index, m_min_eps);
                    if (neighbors.size() >= m_min_points) {
                        this->expand_border_backward(data, max_index, clusters, cluster);
                        this->expand_cluster(data, max_index, neighbors, clusters, cluster);
                        this->expand_border_forward(data, max_index, clusters, cluster);
                        ++cluster;
                    } else {
                        clusters.at(max_index)[-1] = 1.0;
                    }
                    ++max_index;
                }
                return clusters;
            }
        };

        template <class T1, class T2>
        class DBPack: public BaseDBPack<T1, T2> {
        private:
            T1 m_min_eps;
            T1 m_max_eps;
            T2 m_min_points;
            T2 m_max_points;
        protected:

            void expand_cluster(const std::vector<T1> &data) {

            }

            void expand_border_forward(const std::vector<T1> &data) {

            }

            void expand_border_backward(const std::vector<T1> &data) {

            }

        public:
            DBPack(const T1 min_eps, const T1 max_eps, const T2 min_points, const T2 max_points): BaseDBPack<T1, T2>(min_eps, min_points) {
                assert(min_eps > 0);
                assert(max_eps >= min_eps);
                assert(min_points > 0);
                assert(max_points >= min_points);
                m_min_eps = min_eps;
                m_max_eps = max_eps;
                m_min_points = min_points;
                m_max_points = max_points;
            }
            ~DBPack() {};

            std::vector<std::map<T2, T1> > predict(std::vector<T1> data) {
                const std::size_t sample_count = data.size();
                std::vector<std::map<T2, T1> > clusters(sample_count);
                return clusters;
            }
        };
    }
}
