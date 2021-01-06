#ifndef MOVING_H
#define MOVING_H

#include "dbscan.cpp"
#include <set>
#include <map>
#include <algorithm>
#include <tuple>

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

        struct ConvoyCandidate {
            std::vector<size_t> indices = {};
            bool is_assigned = false;
            size_t start_time = 0;
            size_t end_time = 0;
        };


        template <typename T>
        class CMC {
            /*
            An implementation of the Coherence Moving Cluster (CMC) algorithm used to
            identify convoys/flocks from trajectory data.  A convoy/flock is defined
            as a group of objects that move together (within some minimum distance
            of each other) over time for more than some minimum duration of time.

            Examples: commuters along a common route, migrating animals, etc

            k: minimum lifetime constraint
            m: minimum number of common objects

            Original paper: Discovery of Convoys in Trajectory Databases
            https://arxiv.org/pdf/1002.0963.pdf
            */

            private:
                density::DBSCAN<T> m_estimator;
                unsigned int m_k;
                unsigned int m_m;

            public:
                CMC(density::DBSCAN<T>& estimator, const unsigned int k, const unsigned int m) {
                    assert(k > 0);
                    assert(m > 0);
                    m_estimator = estimator;
                    m_k = k;
                    m_m = m;
                }
                virtual ~CMC() {};

                std::tuple<std::vector<std::vector<size_t> >, std::vector<size_t>, std::vector<size_t> > predict(const std::vector< std::vector<std::vector<T> > > &data) {
                    const size_t columns = data.front().size();
                    const size_t data_size = data.size();

                    std::vector<std::vector<size_t> > indices;
                    std::vector<size_t> start_times;
                    std::vector<size_t> end_times;
                    std::vector<ConvoyCandidate> convoy_candidates;

                    std::set<int> used_clusters;
                    std::set<int> previous_unique_clusters;
                    std::vector<int> previous_clusters;
                    for (size_t column = 0; column < columns; ++column) {
                        std::vector<std::vector<T> > column_data = std::vector<std::vector<T> >(data_size);
                        for (size_t i = 0; i < data_size; ++i) {
                            column_data[i] = data[i][column];
                        }
                        std::vector<int> clusters = m_estimator.predict(column_data);
                        std::vector<ConvoyCandidate> current_candidates;
                        std::set<int> unique_clusters;
                        for (const int &i: clusters) {
                            if (i < 0) { continue; }
                            unique_clusters.insert(i);
                        }
                        std::vector<ConvoyCandidate> potential_convoys(unique_clusters.size());
                        for(size_t i = 0; i < data_size; ++i) {
                            int cluster = clusters[i];
                            if (cluster < 0) {
                                continue;
                            }
                            potential_convoys[cluster].indices.push_back(i);
                        }

                        for (size_t i = 0; i < convoy_candidates.size(); ++i) {
                            convoy_candidates[i].is_assigned = false;
                            for (size_t j = 0; j < potential_convoys.size(); ++j) {
                                ConvoyCandidate potential_convoy = potential_convoys[j];
                                std::vector<size_t> cluster_candidate_intersections = vector_intersection(potential_convoy.indices, convoy_candidates[i].indices);
                                if (cluster_candidate_intersections.size() < m_m) {
                                    continue;
                                }
                                convoy_candidates[i].indices = cluster_candidate_intersections;
                                convoy_candidates[i].end_time = column;
                                convoy_candidates[i].is_assigned = true;
                                potential_convoys[j].is_assigned = true;
                                current_candidates.push_back(convoy_candidates[i]);
                            }
                            // creating new candidates
                            size_t lifetime = convoy_candidates[i].end_time - convoy_candidates[i].start_time;
                            if ((!convoy_candidates[i].is_assigned || column == columns - 1) && lifetime >= m_k) {
                                indices.push_back(convoy_candidates[i].indices);
                                start_times.push_back(convoy_candidates[i].start_time);
                                end_times.push_back(convoy_candidates[i].end_time);
                            }
                        }
                        for (size_t i = 0; i < potential_convoys.size(); ++i) {
                            if (potential_convoys[i].is_assigned) {
                                continue;
                            }
                            potential_convoys[i].start_time = column;
                            potential_convoys[i].end_time = column;
                            current_candidates.push_back(potential_convoys[i]);
                        }
                        convoy_candidates = current_candidates;
                    }
                    std::tuple<std::vector<std::vector<size_t>>, std::vector<size_t>, std::vector<size_t> > output = std::tie(indices, start_times, end_times);
                    return output;
                }
        };
    }
}

#endif /* MOVING_H */
