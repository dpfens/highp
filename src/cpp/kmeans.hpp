#ifndef KMEANS_H
#define KMEANS_H

#include <stdlib.h>
#include <tuple>
#include <vector>
#include <algorithm>
#include <unordered_map>


namespace clustering {

    template <typename T>
    class KMeansContiguous {
        private:
            long int m_k;
            long int m_max_iterations;
            T m_tolerance;
            long int m_dimensions;
            T (* m_distance)(const T*, const T*, long int);
            
            long double * m_sums;
            size_t * m_counts;
            T * m_new_centroids;
            bool m_buffers_allocated;

            void allocate_buffers() {
                if (!m_buffers_allocated) {
                    m_sums = (long double *) malloc(sizeof(long double) * m_k * m_dimensions);
                    m_counts = (size_t *) malloc(sizeof(size_t) * m_k);
                    m_new_centroids = (T *) malloc(sizeof(T) * m_k * m_dimensions);
                    m_buffers_allocated = true;
                }
            }

            void deallocate_buffers() {
                if (m_buffers_allocated) {
                    free(m_sums);
                    free(m_counts);
                    free(m_new_centroids);
                    m_buffers_allocated = false;
                }
            }

            void initialize_random_centroids(T* data, size_t dataLength, T* centroids) { 
                size_t i = 0;
                size_t dataPoints = dataLength / m_dimensions;

                srand(time(NULL)); 
                for (i = 0; i < m_k; ++i) {
                    // initialize centroid to a random point from the provided data
                    long int random_seed = rand() % (dataPoints + 1);
                    for (size_t j = 0; j < m_dimensions; ++j) {
                        centroids[i * m_dimensions + j] = data[random_seed * m_dimensions + j];
                    }
                }
            }

            void initialize_kpp_centroids(T* data, size_t dataLength, T *centroids) {
                size_t dataPoints = dataLength / m_dimensions;
                size_t i = 0;

                T ** centroidIndices = (T **) malloc(sizeof(T *) * m_k);
                srand(time(NULL)); 

                //set first seed
                size_t random_seed = rand() % (dataPoints + 1);
                centroidIndices[0] = &data[random_seed * m_dimensions];
                
                for (i = 0; i < m_dimensions; ++i) {
                    centroids[i] = data[random_seed * m_dimensions + i];
                }
                long int current_centroid = 1;

                while (current_centroid < m_k) {
                    T maxMinDistance = std::numeric_limits<T>::min();
                    long int maxCentroidIndex = 0;
                    for (size_t j = 0; j < dataPoints; ++j) {
                        // check if already selected as a centroid
                        bool isSelected = false;
                        for (size_t k = 0; k < current_centroid; ++k) {
                            if (&data[j * m_dimensions] == centroidIndices[k]) {
                                isSelected = true;
                                break;
                            }
                        }
                        if (isSelected) {
                            continue;
                        }
                        /* end Check if already selected */
                        
                        T currentMinDistance = std::numeric_limits<T>::max();
                        for (long int k = 0; k < current_centroid; ++k) {
                            T potentialDistance = m_distance(&centroids[k * m_dimensions], &data[j * m_dimensions], m_dimensions);
                            if (potentialDistance < currentMinDistance) {
                                currentMinDistance = potentialDistance;
                            }
                        }

                        if (currentMinDistance > maxMinDistance) {
                            maxMinDistance = currentMinDistance;
                            maxCentroidIndex = j;
                        }
                    }

                    centroids[current_centroid * m_dimensions] = data[maxCentroidIndex * m_dimensions];
                    centroidIndices[current_centroid] = &data[maxCentroidIndex * m_dimensions];
                    ++current_centroid;
                }
                free(centroidIndices);
            }

            T update_centroids(T* data, size_t dataLength, T* centroids, long int * clusters) {
                size_t dataPoints = dataLength / m_dimensions;

                allocate_buffers();

                for (size_t cluster = 0; cluster < m_k; ++cluster) {
                    m_counts[cluster] = 0;
                    for (size_t dimension = 0; dimension < m_dimensions; ++dimension) {
                        m_sums[cluster * m_dimensions + dimension] = 0.0;
                    }
                }

                for (size_t i = 0; i < dataPoints; ++i) {
                    const long int cluster = clusters[i];
                    m_counts[cluster]++;
                    
                    const size_t data_offset = i * m_dimensions;
                    const size_t sum_offset = cluster * m_dimensions;
                    for (size_t j = 0; j < m_dimensions; ++j) {
                        m_sums[sum_offset + j] += data[data_offset + j];
                    }
                }

                for (size_t i = 0; i < m_k; ++i) {
                    if (m_counts[i] > 0) {
                        const T count_inv = 1.0 / (T)m_counts[i];
                        const size_t centroid_offset = i * m_dimensions;
                        for (size_t j = 0; j < m_dimensions; ++j) {
                            m_new_centroids[centroid_offset + j] = (T)(m_sums[centroid_offset + j] * count_inv);
                        }
                    } else {
                        size_t random_seed = rand() % (dataPoints + 1);
                        const size_t centroid_offset = i * m_dimensions;
                        const size_t data_offset = random_seed * m_dimensions;
                        for (size_t j = 0; j < m_dimensions; ++j) {
                            m_new_centroids[centroid_offset + j] = data[data_offset + j];
                        }
                    }
                }

                T changes = 0.0;
                for (size_t i = 0; i < m_k; ++i) {
                    const size_t centroid_offset = i * m_dimensions;
                    T distance = m_distance(&centroids[centroid_offset], &m_new_centroids[centroid_offset], m_dimensions);
                    changes += distance;
                    for (size_t j = 0; j < m_dimensions; ++j) {
                        centroids[centroid_offset + j] = m_new_centroids[centroid_offset + j];
                    }
                }
                return changes;
            }

            long int update_clusters(T* data, size_t dataLength, T* centroids, long int * clusters) {
                size_t dataPoints = dataLength / m_dimensions;
                long int assignment_changes = 0;

                for (size_t i = 0; i < dataPoints; ++i) {
                    const T* sample = &data[i * m_dimensions];
                    long int closest_centroid = 0;
                    T closest_centroid_distance = std::numeric_limits<T>::max();

                    for (size_t j = 0; j < m_k; ++j) {
                        const T* centroid = &centroids[j * m_dimensions];
                        const T distance = this->m_distance(sample, centroid, m_dimensions);
                        if (distance < closest_centroid_distance) {
                            closest_centroid_distance = distance;
                            closest_centroid = j;
                        }
                    }
                    if (clusters[i] != closest_centroid) {
                        clusters[i] = closest_centroid;
                        ++assignment_changes;
                    }
                }
                return assignment_changes;
            }

        public:
            KMeansContiguous(const long int k, const long int max_iterations, const T tolerance, const long int dimensions, T (* distance_func)(const T*, const T*, long int)) {
                m_k = k;
                m_max_iterations = max_iterations;
                m_tolerance = tolerance;
                m_dimensions = dimensions;
                m_distance = distance_func;
                m_buffers_allocated = false;
                m_sums = nullptr;
                m_counts = nullptr;
                m_new_centroids = nullptr;
            }

            ~KMeansContiguous() {
                deallocate_buffers();
            }

            void setK(const long int k) {
                if (k != m_k) {
                    deallocate_buffers();
                    m_k = k;
                }
            }

            long int getK() {
                return this->m_k;
            }

            void setDimensions(const long int dimensions) {
                if (dimensions != m_dimensions) {
                    deallocate_buffers();
                    m_dimensions = dimensions;
                }
            }

            long int getDimensions() {
                return this->m_dimensions;
            }

            void setMaxIterations(const long int maxIterations) {
                this->m_max_iterations = maxIterations;
            }

            long int getMaxIterations() {
                return this->m_max_iterations;
            }

            void setTolerance(const T tolerance) {
                this->m_tolerance = tolerance;
            }

            T getTolerance() {
                return this->m_tolerance;
            }

            std::tuple<T * , long int * > predict(T* data, size_t length) {
                long int pointCount = length / m_dimensions;
                long int * clusters = (long int *) malloc(sizeof(long int) * pointCount);
                T* centroids = (T *) malloc(sizeof(T) * m_dimensions * m_k);
                long int current_iteration = 0;
                T centroid_changes = m_tolerance;
                long int assignment_changes = 1; // Initialize to non-zero to enter loop

                initialize_kpp_centroids(data, length, centroids);

                while (current_iteration < m_max_iterations && 
                       centroid_changes >= m_tolerance && 
                       assignment_changes > 0) {
                    ++current_iteration;
                    assignment_changes = update_clusters(data, length, centroids, clusters);

                    if (assignment_changes == 0) {
                        break;
                    }
                    centroid_changes = update_centroids(data, length, centroids, clusters);
                }
                std::tuple<T *, long int *> output = std::tie(centroids, clusters);
                return output;
            }
    };


    template <typename T>
    class KMeans {
    private:
        long int m_k;
        long int m_max_iterations;
        T m_tolerance;
        T (* m_distance)(std::vector<T>, std::vector<T>);

        void initialize_random_centroids(std::vector<std::vector<T> > &data, std::vector<std::vector<T> > &centroids) {
            size_t sample_count = data.size();
            size_t centroid_count = centroids.size();
            size_t dimensions = data[0].size();
            size_t i = 0;

            for (i = 0; i < centroid_count; ++i) {
                centroids[i] = std::vector<T>(dimensions);
                T random_seed = rand() % (sample_count + 1);
                for (size_t j = 0; j < dimensions; ++j) {
                    centroids[i][j] = data[random_seed][j];
                }
            }
        }

        void initialize_kpp_centroids(std::vector<std::vector<T> > &data, std::vector<std::vector<T> > &centroids) {
            size_t sample_count = data.size();
            size_t centroid_count = centroids.size();
            size_t dimensions = data[0].size();
            size_t i = 0;

            for (i = 0; i < centroid_count; ++i) {
                centroids[i] = std::vector<T>(dimensions);
            }

            //set first seed
            size_t random_seed = rand() % (sample_count + 1);
            #pragma omp parallel for private(i) shared(centroids, data)
            for (i = 0; i < dimensions; ++i) {
                centroids[0][i] = data[random_seed][i];
            }
            std::vector<T> last_centroid = centroids[0];
            long int current_centroid = 1;

            while (current_centroid < m_k) {

                std::vector<T> distances;
                for (size_t j = 0; j < sample_count; ++j) {
                    std::vector<T> potential_point = data[j];
                    T current_min_distance = 99999;
                    for (long int k = 0; k < current_centroid; ++k) {
                        T potential_distance = m_distance(centroids[k], potential_point);
                        if (potential_distance < current_min_distance) {
                            current_min_distance = potential_distance;
                        }
                    }
                    distances.push_back(current_min_distance);
                }

                size_t last_centroid_index = std::distance(distances.begin(), std::max_element(distances.begin(), distances.end()));
                last_centroid = data[last_centroid_index];
                centroids[current_centroid] = last_centroid;
                ++current_centroid;
            }
        }

        T update_centroids(std::vector<std::vector<T> > &data, std::vector<std::vector<T> > &centroids, std::vector<long int> &clusters) {
            size_t centroid_count = centroids.size();
            size_t sample_count = data.size();
            size_t i = 0;
            size_t dimensions = data[0].size();

            std::vector<std::vector<T> > sums(centroid_count);
            std::vector<std::vector<size_t> > counts(centroid_count);
            std::vector<std::vector<T> > new_centroids(centroid_count);

            for (i = 0; i < centroid_count; ++i) {
                new_centroids[i] = std::vector<T>(dimensions);
                sums[i] = std::vector<T>(dimensions);
                counts[i] = std::vector<size_t>(dimensions);
            }

            #pragma omp parallel for private(i) shared(data, centroids)
            for (i = 0; i < dimensions; ++i) {
                for (size_t j = 0; j < sample_count; ++j) {
                    std::vector<T> sample = data[j];
                    long int cluster = clusters[j];
                    size_t count = counts[cluster][i];
                    counts[cluster][i] = count + 1;
                    sums[cluster][i] += sample[i];
                }
            }

            #pragma omp parallel for private(i) shared(centroids)
            for (i = 0; i < centroid_count; ++i) {
                for (size_t j = 0; j < dimensions; ++j) {
                    T sum = sums[i][j];
                    size_t count = counts[i][j];
                    new_centroids[i][j] = sum / (T)count;
                }
            }

            T changes = 0.0;
            for (i = 0; i < centroid_count; ++i) {
                T distance = m_distance(centroids[i], new_centroids[i]);
                changes += distance;
                for (size_t j = 0; j < dimensions; ++j) {
                    centroids[i][j] = new_centroids[i][j];
                }
            }

            return changes;
        }

        long int update_clusters(std::vector<std::vector<T> > &data, std::vector<std::vector<T> > &centroids, std::vector<long int> &clusters) {
            size_t centroid_count = centroids.size();
            size_t sample_count = data.size();
            size_t i = 0;

            long int assignment_changes = 0;

            T distance = 0.0;
            long int closest_centroid = 0;
            T closest_centroid_distance = 9999999;
            for (i = 0; i < sample_count; ++i) {
                std::vector<T>& sample = data[i];
                closest_centroid = 0;
                closest_centroid_distance = 9999999;
                for (size_t j = 0; j < centroid_count; ++j) {
                    std::vector<T>& centroid = centroids[j];
                    distance = this->m_distance(sample, centroid);
                    if (distance < closest_centroid_distance) {
                        closest_centroid_distance = distance;
                        closest_centroid = j;
                    }
                }
                if (clusters[i] != closest_centroid) {
                    clusters[i] = closest_centroid;
                    #pragma omp critical
                    ++assignment_changes;
                }
            }
            return assignment_changes;
        }

    public:
        KMeans(const long int k, const long int max_iterations, const T tolerance, T (* distance_func)(std::vector<T>, std::vector<T>)) {
            m_k = k;
            m_max_iterations = max_iterations;
            m_tolerance = tolerance;
            m_distance = distance_func;
        }

        void setK(const long int k) {
            this->m_k = k;
        }

        long int getK() {
            return this->m_k;
        }

        void setMaxIterations(const long int maxIterations) {
            this->m_max_iterations = maxIterations;
        }

        long int getMaxIterations() {
            return this->m_max_iterations;
        }

        void setTolerance(const T tolerance) {
            this->m_tolerance = tolerance;
        }

        T getTolerance() {
            return this->m_tolerance;
        }

        std::tuple<std::vector<std::vector<T> >, std::vector<long int> > predict(std::vector<std::vector<T> > &data) {

            size_t sample_size = data.size();
            std::vector<long int> clusters(sample_size);
            std::vector<std::vector<T> > centroids(m_k);
            long int current_iteration = 0;
            T centroid_changes = m_tolerance;
            long int assignment_changes = m_tolerance;

            initialize_random_centroids(data, centroids);
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
        KMedian(const long int k, const long int max_iterations, const T tolerance, T (* distance_func)(std::vector<T>, std::vector<T>)): KMeans<T>(k, max_iterations, tolerance, distance_func) {
            m_k = k;
            m_max_iterations = max_iterations;
            m_tolerance = tolerance;
            m_distance = distance_func;
        }

    private:
        long int m_k;
        long int m_max_iterations;
        T m_tolerance;
        T (* m_distance)(std::vector<T>, std::vector<T>);

        void initialize_random_centroids(std::vector<std::vector<T> > &data, std::vector<std::vector<T> > &centroids) {
            size_t sample_count = data.size();
            size_t centroid_count = centroids.size();
            size_t dimensions = data[0].size();
            size_t i = 0;

            #pragma omp parallel for private(i)
            for (i = 0; i < dimensions; ++i) {

                std::vector<T> values(sample_count);
                for(size_t j = 0; j < sample_count; ++j) {
                    values.push_back(data[j][i]);
                }

                T dimension_median = median(values);
                for(size_t j = 0; j < centroid_count; ++j) {
                    centroids[j][i] = dimension_median;
                }
            }
        }

        T median(std::vector<T> data) {
            std::sort(data.begin(), data.end());
            return data[data.size() / 2];
        }

        T update_centroids(std::vector<std::vector<T> > &data, std::vector<std::vector<T> > &centroids, std::vector<int> &clusters) {
            size_t centroid_count = centroids.size();
            size_t sample_count = data.size();
            size_t dimensions = centroids[0].size();
            size_t i = 0;

            std::vector<std::vector<T> > new_centroids(centroid_count);

            for (i = 0; i < centroid_count; ++i) {
                new_centroids[i] = std::vector<T>(dimensions);
            }

            #pragma omp parallel for private(i)
            for (i = 0; i < dimensions; ++i) {
                std::vector<std::vector<T> > centroid_values(centroid_count);

                for(size_t j = 0; j < sample_count; ++j) {
                    size_t cluster = clusters[j];
                    centroid_values[cluster].push_back(data[j][i]);
                }

                for(size_t j = 0; j < centroid_count; ++j) {
                    T dimension_median = median(centroid_values[j]);
                    centroids[j][i] = dimension_median;
                }
            }

            T changes = 0.0;
            #pragma omp parallel for private(i)
            for (i = 0; i < centroid_count; ++i) {
                T distance = m_distance(centroids[i], new_centroids[i]);
                #pragma omp critical
                changes += distance;
            }

            return changes;
        }
    };

    template <typename T>
    class KMode: public KMeans<T> {
    public:
        KMode(const long int k, const long int max_iterations, const T tolerance, T (* distance_func)(std::vector<T>, std::vector<T>)): KMeans<T>(k, max_iterations, tolerance, distance_func) {
            m_k = k;
            m_max_iterations = max_iterations;
            m_tolerance = tolerance;
            m_distance = distance_func;
        }

        static T dissimilarity_func(std::vector<T> point1, std::vector<T> point2) {
            size_t dimensions = point1.size();
            T output = 0;
            for (size_t i = 0; i < dimensions; ++i) {
                if(point1[i] != point2[i]) {
                    ++output;
                }
            }
            return output;
        }

    private:
        long int m_k;
        long int m_max_iterations;
        T m_tolerance;
        T (* m_distance)(std::vector<T>, std::vector<T>);

        T update_centroids(std::vector<std::vector<T> > &data, std::vector<std::vector<T> > &centroids, std::vector<long int> &clusters) {
            size_t centroid_count = centroids.size();
            size_t sample_count = data.size();
            size_t dimensions = centroids[0].size();
            size_t i = 0;
            T changes = 0.0;

            std::vector<std::vector<T> > new_centroids(centroid_count);

            for (i = 0; i < centroid_count; ++i) {
                new_centroids[i] = std::vector<T>(dimensions);
            }

            #pragma omp parallel for private(i)
            for (i = 0; i < dimensions; ++i) {
                std::unordered_map<long int, std::unordered_map<T, size_t> > cluster_dimension_values;

                for(size_t j = 0; j < sample_count; ++j) {
                    T dimension_category = data[j][i];
                    long int cluster = clusters[j];
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
                    centroids[j][i] = current_max_category;
                    ++changes;
                }
            }
            return changes;
        }
    };
}

#endif /* KMEANS_H */
