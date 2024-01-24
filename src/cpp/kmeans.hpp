#ifndef KMEANS_H
#define KMEANS_H

#include <stdlib.h>
#include <tuple>
#include <vector>
#include <algorithm>
#include <unordered_map>


namespace clustering {

    template <typename T>
    class KMeans {
    private:
        long int m_k;
        long int m_max_iterations;
        double m_tolerance;
        double (* m_distance)(std::vector<T>, std::vector<T>);

        void initialize_random_centroids(std::vector<std::vector<T> > &data, std::vector<std::vector<T> > &centroids) {
            size_t sample_count = data.size();
            size_t centroid_count = centroids.size();
            size_t dimensions = data.at(0).size();
            size_t i = 0;

            for (i = 0; i < centroid_count; ++i) {
                centroids.at(i) = std::vector<T>(dimensions);
                T random_seed = rand() % (sample_count + 1);
                for (size_t j = 0; j < dimensions; ++j) {
                    centroids.at(i).at(j) = data.at(random_seed).at(j);
                }
            }
        }

        void initialize_kpp_centroids(std::vector<std::vector<T> > &data, std::vector<std::vector<T> > &centroids) {
            size_t sample_count = data.size();
            size_t centroid_count = centroids.size();
            size_t dimensions = data.at(0).size();
            size_t i = 0;

            for (i = 0; i < centroid_count; ++i) {
                centroids.at(i) = std::vector<T>(dimensions);
            }

            std::vector<std::vector<double> > matrix(sample_count);
            for (i = 0; i < sample_count; ++i) {
                matrix.at(i) = std::vector<double>(sample_count);
            }

            #pragma omp parallel for private(i) shared(data, matrix)
            for (i = 0; i < sample_count; ++i) {
                for (size_t j = i; j < sample_count; ++j) {
                    double distance;
                    if (i == j) {
                        distance = 0;
                    } else {
                        distance = m_distance(data.at(i), data.at(j));
                    }
                    matrix.at(i).at(j) = distance;
                    matrix.at(j).at(i) = distance;
                }
            }

            //set first seed
            T random_seed = rand() % (sample_count + 1);
            #pragma omp parallel for private(i) shared(centroids, data)
            for (i = 0; i < dimensions; ++i) {
                centroids.at(0).at(i) = data.at(random_seed).at(i);
            }
            std::vector<T> last_centroid = centroids.at(0);
            long int current_centroid = 1;

            while (current_centroid < m_k) {

                std::vector<double> distances;
                for (size_t j = 0; j < sample_count; ++j) {
                    std::vector<T> potential_point = data.at(j);
                    double current_min_distance = 99999;
                    for (long int k = 0; k < current_centroid; ++k) {
                        double potential_distance = m_distance(centroids.at(k), potential_point);
                        if (potential_distance < current_min_distance) {
                            current_min_distance = potential_distance;
                        }
                    }
                    distances.push_back(current_min_distance);
                }

                size_t last_centroid_index = std::distance(distances.begin(), std::max_element(distances.begin(), distances.end()));
                last_centroid = data.at(last_centroid_index);
                centroids.at(current_centroid) = last_centroid;
                ++current_centroid;
            }
        }

        double update_centroids(std::vector<std::vector<T> > &data, std::vector<std::vector<T> > &centroids, std::vector<long int> &clusters) {
            size_t centroid_count = centroids.size();
            size_t sample_count = data.size();
            size_t i = 0;
            size_t dimensions = data.at(0).size();

            std::vector<std::vector<size_t> > sums(centroid_count);
            std::vector<std::vector<size_t> > counts(centroid_count);
            std::vector<std::vector<T> > new_centroids(centroid_count);

            for (i = 0; i < centroid_count; ++i) {
                new_centroids.at(i) = std::vector<T>(dimensions);
                sums.at(i) = std::vector<size_t>(dimensions);
                counts.at(i) = std::vector<size_t>(dimensions);
            }

            #pragma omp parallel for private(i) shared(data, centroids)
            for (i = 0; i < dimensions; ++i) {
                for (size_t j = 0; j < sample_count; ++j) {
                    std::vector<T> sample = data.at(j);
                    long int cluster = clusters.at(j);
                    size_t count = counts.at(cluster).at(i);
                    counts.at(cluster).at(i) = count + 1;
                    sums.at(cluster).at(i) += sample.at(i);
                }
            }

            #pragma omp parallel for private(i) shared(centroids)
            for (i = 0; i < centroid_count; ++i) {
                for (size_t j = 0; j < dimensions; ++j) {
                    size_t sum = sums.at(i).at(j);
                    size_t count = counts.at(i).at(j);
                    new_centroids.at(i).at(j) = sum / count;
                }
            }

            double changes = 0.0;
            for (i = 0; i < centroid_count; ++i) {
                double distance = m_distance(centroids.at(i), new_centroids.at(i));
                changes += distance;
                for (size_t j = 0; j < dimensions; ++j) {
                    centroids.at(i).at(j) = new_centroids.at(i).at(j);
                }
            }

            return changes;
        }

        long int update_clusters(std::vector<std::vector<T> > &data, std::vector<std::vector<T> > &centroids, std::vector<long int> &clusters) {
            size_t centroid_count = centroids.size();
            size_t sample_count = data.size();
            size_t i = 0;

            long int assignment_changes = 0;

            #pragma omp parallel for private(i) shared(data)
            for (i = 0; i < sample_count; ++i) {
                std::vector<T> sample = data.at(i);
                long int closest_centroid = 0;
                double closest_centroid_distance = 9999999;
                for (size_t j = 0; j < centroid_count; ++j) {
                    std::vector<T> centroid = centroids.at(j);
                    double distance = this->m_distance(sample, centroid);
                    if (distance < closest_centroid_distance) {
                        closest_centroid_distance = distance;
                        closest_centroid = j;
                    }
                }
                if (clusters.at(i) != closest_centroid) {
                    clusters.at(i) = closest_centroid;
                    #pragma omp critical
                    ++assignment_changes;
                }
            }
            return assignment_changes;
        }

    public:
        KMeans(const long int k, const long int max_iterations, const double tolerance, double (* distance_func)(std::vector<T>, std::vector<T>)) {
            m_k = k;
            m_max_iterations = max_iterations;
            m_tolerance = tolerance;
            m_distance = distance_func;
        }

        std::tuple<std::vector<std::vector<T> >, std::vector<long int> > predict(std::vector<std::vector<T> > &data) {

            size_t sample_size = data.size();
            std::vector<long int> clusters(sample_size);
            std::vector<std::vector<T> > centroids(m_k);
            long int current_iteration = 0;
            double centroid_changes = m_tolerance;
            long int assignment_changes = m_tolerance;

            initialize_kpp_centroids(data, centroids);
            while (current_iteration < m_max_iterations && centroid_changes >= m_tolerance) {
                ++current_iteration;
                assignment_changes = update_clusters(data, centroids, clusters);
                centroid_changes = update_centroids(data, centroids, clusters);
            }
            std::tuple<std::vector<std::vector<T> >, std::vector<long int> > output = std::tie(centroids, clusters);
            return output;
        }
    };

    template <typename T>
    class KMedian: public KMeans<T> {
    public:
        KMedian(const long int k, const long int max_iterations, const double tolerance, double (* distance_func)(std::vector<T>, std::vector<T>)): KMeans<T>(k, max_iterations, tolerance, distance_func) {
            m_k = k;
            m_max_iterations = max_iterations;
            m_tolerance = tolerance;
            m_distance = distance_func;
        }

    private:
        long int m_k;
        long int m_max_iterations;
        double m_tolerance;
        double (* m_distance)(std::vector<T>, std::vector<T>);

        void initialize_random_centroids(std::vector<std::vector<T> > &data, std::vector<std::vector<T> > &centroids) {
            size_t sample_count = data.size();
            size_t centroid_count = centroids.size();
            size_t dimensions = data.at(0).size();
            size_t i = 0;

            #pragma omp parallel for private(i)
            for (i = 0; i < dimensions; ++i) {

                std::vector<T> values(sample_count);
                for(size_t j = 0; j < sample_count; ++j) {
                    values.push_back(data.at(j).at(i));
                }

                T dimension_median = median(values);
                for(size_t j = 0; j < centroid_count; ++j) {
                    centroids.at(j).at(i) = dimension_median;
                }
            }
        }

        T median(std::vector<T> data) {
            std::sort(data.begin(), data.end());
            return data.at(data.size() / 2);
        }

        double update_centroids(std::vector<std::vector<T> > &data, std::vector<std::vector<T> > &centroids, std::vector<int> &clusters) {
            size_t centroid_count = centroids.size();
            size_t sample_count = data.size();
            size_t dimensions = centroids.at(0).size();
            size_t i = 0;

            std::vector<std::vector<T> > new_centroids(centroid_count);

            for (i = 0; i < centroid_count; ++i) {
                new_centroids.at(i) = std::vector<T>(dimensions);
            }

            #pragma omp parallel for private(i)
            for (i = 0; i < dimensions; ++i) {
                std::vector<std::vector<T> > centroid_values(centroid_count);

                for(size_t j = 0; j < sample_count; ++j) {
                    size_t cluster = clusters.at(j);
                    centroid_values.at(cluster).push_back(data.at(j).at(i));
                }

                for(size_t j = 0; j < centroid_count; ++j) {
                    T dimension_median = median(centroid_values.at(j));
                    centroids.at(j).at(i) = dimension_median;
                }
            }

            double changes = 0.0;
            #pragma omp parallel for private(i)
            for (i = 0; i < centroid_count; ++i) {
                double distance = m_distance(centroids.at(i), new_centroids.at(i));
                #pragma omp critical
                changes += distance;
            }

            return changes;
        }
    };

    template <typename T>
    class KMode: public KMeans<T> {
    public:
        KMode(const long int k, const long int max_iterations, const double tolerance, double (* distance_func)(std::vector<T>, std::vector<T>)): KMeans<T>(k, max_iterations, tolerance, distance_func) {
            m_k = k;
            m_max_iterations = max_iterations;
            m_tolerance = tolerance;
            m_distance = distance_func;
        }

        static double dissimilarity_func(std::vector<T> point1, std::vector<T> point2) {
            size_t dimensions = point1.size();
            double output = 0;
            for (size_t i = 0; i < dimensions; ++i) {
                if(point1.at(i) != point2.at(i)) {
                    ++output;
                }
            }
            return output;
        }

    private:
        long int m_k;
        long int m_max_iterations;
        double m_tolerance;
        double (* m_distance)(std::vector<T>, std::vector<T>);

        double update_centroids(std::vector<std::vector<T> > &data, std::vector<std::vector<T> > &centroids, std::vector<long int> &clusters) {
            size_t centroid_count = centroids.size();
            size_t sample_count = data.size();
            size_t dimensions = centroids.at(0).size();
            size_t i = 0;
            double changes = 0.0;

            std::vector<std::vector<T> > new_centroids(centroid_count);

            for (i = 0; i < centroid_count; ++i) {
                new_centroids.at(i) = std::vector<T>(dimensions);
            }

            #pragma omp parallel for private(i)
            for (i = 0; i < dimensions; ++i) {
                std::unordered_map<long int, std::unordered_map<T, size_t> > cluster_dimension_values;

                for(size_t j = 0; j < sample_count; ++j) {
                    T dimension_category = data.at(j).at(i);
                    long int cluster = clusters.at(j);
                    if (cluster_dimension_values[j].find(dimension_category) == cluster_dimension_values[j].end()) {
                        cluster_dimension_values[j][dimension_category] = 0;
                    }
                    ++cluster_dimension_values[j][dimension_category];
                }

                size_t j = 0;
                for(j = 0; j < centroid_count; ++j) {
                    std::unordered_map<T, size_t> cluster_categories = cluster_dimension_values[j];
                    T current_max_category = (cluster_categories.cbegin())->first;
                    size_t current_max = (cluster_categories.cbegin())->second;
                    for(auto it = cluster_dimension_values.cbegin(); it != cluster_dimension_values.cend(); ++it ) {
                        if (it->second > current_max) {
                            current_max_category = it->first;
                            current_max = it->second;

                        }
                    }
                    centroids.at(j).at(i) = current_max_category;
                    ++changes;
                }
            }
            return changes;
        }
    };
}

#endif /* KMEANS_H */
