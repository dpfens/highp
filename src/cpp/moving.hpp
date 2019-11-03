#include "dbscan.cpp"
#include <set>
#include <map>
#include <algorithm>

namespace density {

    namespace moving {

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

        template <typename T>
        class MovingDBSCAN {

            private:
                density::DBSCAN<T> m_estimator;
                double m_theta;

                double jaccard(std::vector<size_t> A, std::vector<size_t> B) {
                    std::vector<size_t> numerator_set = vector_intersection(A, B);

                    if (!numerator_set.size()) {
                        return 0.0;
                    }
                    std::vector<size_t> denominator_set = vector_union(A, B);
                    return static_cast<double>(numerator_set.size()) / static_cast<double>(denominator_set.size());
                }

            public:
                MovingDBSCAN(density::DBSCAN<T>& estimator, const double theta) {
                    assert(theta > 0.0);
                    assert(theta <= 1.0);
                    m_estimator = estimator;
                    m_theta = theta;
                }
                virtual ~MovingDBSCAN() {};

                std::vector<std::vector<int> > predict(const std::vector< std::vector<std::vector<T> > > &data) {
                    const size_t columns = data.front().size();
                    const size_t data_size = data.size();

                    std::cout << "Columns: " << columns << ", Rows: " << data_size << "\n";

                    std::vector< std::vector<int> > assignments = std::vector< std::vector<int> >(columns);
                    std::set<int> used_clusters;
                    std::set<int> previous_unique_clusters;
                    std::vector<int> previous_clusters;
                    for (size_t column = 0; column < columns; ++column) {
                        std::vector<std::vector<T> > column_data = std::vector<std::vector<T> >(data_size);
                        for (size_t i = 0; i < data_size; ++i) {
                            column_data[i] = data[i][column];
                        }
                        std::vector<int> clusters = m_estimator.predict(column_data);
                        std::set<int> unique_clusters;
                        for (const int &i: clusters) {
                            if (i < 0) { continue; }
                            unique_clusters.insert(i);
                        }

                        std::cout << "Processing Column #" << column << "\n";

                        if (column == 0) {
                            previous_unique_clusters = unique_clusters;
                            used_clusters = unique_clusters;
                            previous_clusters = clusters;
                            assignments[column] = clusters;
                            continue;
                        }

                        for (const int &cluster: unique_clusters) {
                            std::vector<size_t> cluster_indices;
                            for (size_t j = 0; j < data_size; ++j) {
                                if (clusters[j] == cluster) {
                                    cluster_indices.push_back(j);
                                }
                            }

                            std::map <int, double> similarities;
                            for (const int &previous_cluster: previous_unique_clusters) {
                                std::vector<size_t> previous_cluster_indices;
                                for (size_t j = 0; j < data_size; ++j) {
                                    if (previous_clusters[j] == previous_cluster) {
                                        previous_cluster_indices.push_back(j);
                                    }
                                }
                                double jaccard_similarity = jaccard(cluster_indices, previous_cluster_indices);
                                similarities[previous_cluster] = jaccard_similarity;
                            }

                            int new_cluster_id;
                            if (similarities.size() > 0) {
                                int most_similar_cluster;
                                double most_similar_value = 0.0;
                                for(auto it = similarities.cbegin(); it != similarities.cend(); ++it ) {
                                    if (it ->second > most_similar_value) {
                                        most_similar_cluster = it->first;
                                        most_similar_value = it->second;
                                    }
                                }
                                if (most_similar_value >= m_theta) {
                                    new_cluster_id = most_similar_cluster;
                                } else {
                                    new_cluster_id = used_clusters.size();
                                    used_clusters.insert(new_cluster_id);
                                }
                            } else {
                                new_cluster_id = used_clusters.size();
                                used_clusters.insert(new_cluster_id);
                            }
                            // get most similar cluster, to check if the cluster needs to be merged
                            unique_clusters.erase(cluster);
                            unique_clusters.insert(new_cluster_id);

                            for (size_t j = 0; j < cluster_indices.size(); ++j) {
                                clusters[cluster_indices[j]] = new_cluster_id;
                            }
                        }
                        previous_unique_clusters = unique_clusters;
                        previous_clusters = clusters;
                        assignments[column] = clusters;
                    }

                    std::vector<std::vector<int> > output = std::vector<std::vector<int> >(data_size);
                    for (size_t i = 0; i < data_size; ++i) {
                        std::vector<int> row = std::vector<int>(columns);
                        for (size_t j = 0; j < columns; ++j) {
                            row[j] = assignments[j][i];
                        }
                        output[i] = row;
                    }
                    return output;
                }
        };
    }
}
