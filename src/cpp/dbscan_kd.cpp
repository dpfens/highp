#include <cstdbool>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include "distance.cpp"
#include "kdtree/kdtree.cpp"

namespace density {

    template <typename T>
    class DBSCAN {

    private:
        long double m_epsilon;
        unsigned long int m_min_points;
        long double (* m_distance)(std::vector<T>, std::vector<T>);

        void expand_cluster(std::vector<std::vector<T> > &data, const size_t index, std::vector<size_t> index_neighbors, std::vector<int> &clusters, int cluster_id) {
            std::vector<size_t> seed_neighbors = index_neighbors, n_neighbors, visited;
            visited.push_back(index);
            std::vector<T> neighbor;
            int n_index, n_index_cluster, seed;
            while (!seed_neighbors.empty()) {
                seed = seed_neighbors.front();
                seed_neighbors.erase(seed_neighbors.begin());
                if(std::find(visited.begin(), visited.end(), seed) != visited.end()) {
                    continue;
                }
                visited.push_back(seed);
                neighbor = data.at(seed);
                n_neighbors = neighbors(data, neighbor);
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

        std::vector<size_t> neighbors(std::vector<std::vector<T> > &data, std::vector<T> point) {
            long double distance = 0.0;
            std::vector<size_t> output;
            for(auto it = data.begin(); it != data.end(); ++it) {
                size_t index = std::distance(data.begin(), it);
                distance = m_distance(*it, point);
                if (distance < m_epsilon) {
                    output.push_back(index);
                }
            }
            return output;
        }

        public:
            DBSCAN(const long double epsilon, const unsigned long int min_points, long double (* distance_func)(std::vector<T>, std::vector<T>)) {
                assert(epsilon > 0);
                assert(min_points > 0);
                m_epsilon = epsilon;
                m_min_points = min_points;
                m_distance = distance_func;
            }

            std::vector<int> predict(std::vector<std::vector<T> > &data) {
                const std::size_t sample_count = data.size();
                std::vector<int> clusters(sample_count);

                KDTree<T> tree = KDTree<T>(data);

                for(auto it = clusters.begin(); it != clusters.end(); ++it) {
                    *it = -2;
                }

                int cluster_id = 0;
                int index;
                for(auto it = data.begin(); it != data.end(); ++it) {
                    index = std::distance(data.begin(), it);
                    if (clusters.at(index) != -2) {
                        continue;
                    }

                    std::vector<size_t> point_neighbors = neighbors(data, *it);
                    if (point_neighbors.size() < m_min_points) {
                        clusters.at(index) = -1;
                    }
                    else {
                        clusters.at(index) = cluster_id;
                        expand_cluster(data, index, point_neighbors, clusters, cluster_id);
                        cluster_id += 1;
                     }
                }
                return clusters;
            }
    };

};


int main() {
    long double epsilon = 1.5;
    unsigned long int min_points = 2;
    density::DBSCAN<long double> clf = density::DBSCAN<long double>(epsilon, min_points, euclidean);

    std::vector<std::vector<long double> > data = {{931.0}, {931.0}, {932.0}, {932.0}, {932.0}, {932.0}, {932.0}, {932.0}, {933.0}, {933.0}, {933.0}, {933.0}, {933.0}, {933.0}, {933.0}, {933.0}, {933.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {935.0}, {935.0}, {935.0}, {935.0}, {935.0}, {936.0}, {936.0}, {936.0}, {936.0}, {936.0}, {936.0}, {937.0}, {938.0}, {938.0}, {938.0}, {938.0}, {938.0}, {939.0}, {939.0}, {939.0}, {939.0}, {939.0}, {940.0}, {940.0}, {940.0}, {940.0}, {941.0}, {941.0}, {941.0}, {942.0}, {942.0}, {942.0}, {943.0}, {944.0}, {944.0}, {945.0}, {945.0}, {945.0}, {945.0}, {946.0}, {946.0}, {947.0}, {947.0}, {947.0}, {948.0}, {948.0}, {948.0}, {949.0}, {949.0}, {949.0}, {949.0}, {949.0}, {950.0}, {950.0}, {950.0}, {950.0}, {951.0}, {951.0}, {952.0}, {953.0}, {953.0}, {955.0}, {955.0}, {965.0}, {966.0}, {966.0}, {966.0}, {966.0}, {967.0}, {968.0}, {968.0}, {968.0}, {968.0}, {969.0}, {969.0}, {970.0}, {970.0}, {970.0}, {971.0}, {971.0}, {972.0}, {972.0}, {972.0}, {973.0}, {973.0}, {974.0}, {980.0}, {980.0}, {981.0}, {981.0}, {981.0}, {982.0}, {983.0}, {983.0}, {983.0}, {983.0}, {984.0}, {984.0}, {994.0}, {994.0}, {996.0}, {1002.0}, {1007.0}, {1007.0}, {1007.0}, {1007.0}, {1008.0}, {1009.0}, {1009.0}, {1010.0}, {1028.0}, {1030.0}, {1061.0}, {1078.0}};
    std::vector<int> clusters = clf.predict(data);
    for (auto i = clusters.begin(); i != clusters.end(); ++i) {
        size_t index = std::distance(clusters.begin(), i);
        std::cout << "Index: " << index << " Cluster: " << *i << '\n';
    }
}
