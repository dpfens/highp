#include <cstdbool>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include "distance.cpp"

namespace density {

    template <typename T>
    class DBSCAN {

    private:
        double m_epsilon;
        long int m_min_points;
        double (* m_distance)(std::vector<T>, std::vector<T>);

        void expand_cluster(std::vector<std::vector<T> > &distance_matrix, const int index, std::vector<int> index_neighbors, std::vector<int> &clusters, int cluster_id) {
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
                if (n_neighbors.size() < m_min_points) {
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

        std::vector<int> neighbors(std::vector<std::vector<double> > &distance_matrix, size_t index) {
            double distance;
            std::vector<int> output;
            std::vector<double> row = distance_matrix.at(index);
            for(auto it = row.begin(); it != row.end(); ++it) {
                int index = std::distance(row.begin(), it);
                distance = row.at(index);
                if (distance < m_epsilon) {
                    output.push_back(index);
                }
            }
            return output;
        }

        virtual std::vector<std::vector<double> > calculate_distances(std::vector<std::vector<T> > &data) {
            size_t data_size = data.size(), i = 0, j = 0;
            std::vector<std::vector<double> > output(data_size);
            std::vector<T> point1, point2;
            double dist;

            for(auto i = 0; i < data_size; ++i) {
                std::vector<double> row(data_size);
                output.at(i) = row;
            }
            i = 0;
            for (i = 0; i < data_size; ++i) {
                point1 = data.at(i);
                for(j = i; j < data_size; ++j) {
                    dist = m_distance(point1, data.at(j));
                    output.at(i).at(j) = dist;
                    output.at(j).at(i) = dist;
                }
            }
            return output;
        }

    public:
        DBSCAN(const double epsilon, const long int min_points, double (* distance_func)(std::vector<T>, std::vector<T>)) {
            assert(epsilon > 0);
            assert(min_points > 0);
            m_epsilon = epsilon;
            m_min_points = min_points;
            m_distance = distance_func;
        }

        std::vector<int> predict(std::vector<std::vector<T> > data) {
            const std::size_t sample_count = data.size();
            std::vector<int> clusters(sample_count);

            for(auto it = clusters.begin(); it != clusters.end(); ++it) {
                *it = -2;
            }

            std::vector<std::vector<double> > distance_matrix = this->calculate_distances(data);
            int cluster_id = 0;
            int index;
            for(auto it = data.begin(); it != data.end(); ++it) {
                index = std::distance(data.begin(), it);
                if (clusters.at(index) != -2) {
                    continue;
                }
                std::vector<int> point_neighbors = neighbors(distance_matrix, index);
                if (point_neighbors.size() < m_min_points) {
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

};
