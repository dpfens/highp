#ifndef WASM_KMEANS_H
#define WASM_KMEANS_H

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "../kmeans.cpp"
#include "../distance.hpp"
#include <wasm_simd128.h>

#include "utility.hpp"

namespace wasm {

    namespace cluster {

        struct KResult {
            emscripten::val centroids;
            emscripten::val clusters;
        };

        struct KMeansDirectResult {
            double* centroids;
            long int* clusters;
            long int centroidsLength;
            long int clustersLength;
        };

        template <typename T>
        T ssd(const T* point1, const T* point2, long int dimensions) {
            // Sum of Squared Difference (SSD)
            T distance = 0.0;

            #ifdef __wasm_simd128__
            if constexpr (std::is_same_v<T, float>) {
                v128_t sum = wasm_f32x4_splat(0.0f);
                const int SIMD_BLOCK_SIZE = 4;
                long int i = 0;
                for (; i <= dimensions - SIMD_BLOCK_SIZE; i += SIMD_BLOCK_SIZE) {
                    v128_t vec1 = wasm_f32x4_make(point1[i], point1[i + 1], point1[i + 2], point1[i + 3]);
                    v128_t vec2 = wasm_f32x4_make(point2[i], point2[i + 1], point2[i + 2], point2[i + 3]);
                    v128_t diff = wasm_f32x4_sub(vec1, vec2);
                    v128_t sqr_diff = wasm_f32x4_mul(diff, diff);
                    sum = wasm_f32x4_add(sum, sqr_diff);
                }
                for (; i < dimensions; ++i) {
                    T diff = point2[i] - point1[i];
                    distance += diff * diff;
                }
                distance += wasm_f32x4_extract_lane(sum, 0) + wasm_f32x4_extract_lane(sum, 1) +
                            wasm_f32x4_extract_lane(sum, 2) + wasm_f32x4_extract_lane(sum, 3);
            } else if constexpr (std::is_same_v<T, double>) {
                v128_t sum = wasm_f64x2_splat(0.0);
                const int SIMD_BLOCK_SIZE = 2;
                long int i = 0;
                for (; i <= dimensions - SIMD_BLOCK_SIZE; i += SIMD_BLOCK_SIZE) {
                    v128_t vec1 = wasm_f64x2_make(point1[i], point1[i + 1]);
                    v128_t vec2 = wasm_f64x2_make(point2[i], point2[i + 1]);
                    v128_t diff = wasm_f64x2_sub(vec1, vec2);
                    v128_t sqr_diff = wasm_f64x2_mul(diff, diff);
                    sum = wasm_f64x2_add(sum, sqr_diff);
                }
                for (; i < dimensions; ++i) {
                    T diff = point2[i] - point1[i];
                    distance += diff * diff;
                }
                distance += wasm_f64x2_extract_lane(sum, 0) + wasm_f64x2_extract_lane(sum, 1);
            }
            #else
            for (std::size_t i = 0; i < dimensions; i++) {
                T diff = point2[i] - point1[i];
                distance += (diff * diff);
            }
            #endif

            return distance;
        }

        template <typename T>
        T euclidean(const T* point1, const T* point2, long int dimensions) {
            // Euclidean Distance
            return sqrt(ssd<T>(point1, point2, dimensions));
        }


        // Distance functions as template parameters
        template<typename T>
        struct EuclideanDistance {
            static T compute(const T* a, const T* b, long int dims) {
                return euclidean<T>(a, b, dims);
            }
        };

        template<typename T>
        struct SSDDistance {
            static T compute(const T* a, const T* b, long int dims) {
                return ssd<T>(a, b, dims);
            }
        };

        /**
         * WASM wrapper interface for KMeans Contiguous
         */
        template <typename T, typename DistanceFunc>
        class KMeans {

            public:
                KMeans(const long int k, const long int max_iterations, const T tolerance, long int dimensions) {
                    m_instance = new clustering::KMeansContiguous<T, DistanceFunc>(k, max_iterations, tolerance, dimensions);
                }
                
                ~KMeans() {
                    delete m_instance;
                }

                KResult predict(uintptr_t dataPtr, long int dataLength) {
                    // Cast the pointer to the appropriate type
                    T* data = reinterpret_cast<T*>(dataPtr);
                    
                    return predictInternal(data, dataLength);
                }

                void setK(const long int k) {
                    this->m_instance->setK(k);
                }

                long int getK() {
                    return this->m_instance->getK();
                }

                void setMaxIterations(const long int maxIterations) {
                    this->m_instance->setMaxIterations(maxIterations);
                }

                long int getMaxIterations() {
                    return this->m_instance->getMaxIterations();
                }

                void setTolerance(const T tolerance) {
                    this->m_instance->setTolerance(tolerance);
                }

                T getTolerance() {
                    return this->m_instance->getTolerance();
                }

            private:
                clustering::KMeansContiguous<T, DistanceFunc> * m_instance;

                KResult predictInternal(T* data, long int dataLength) {
                    auto results = this->m_instance->predict(data, dataLength);

                    // Convert data to Javascript
                    long int dimensions = m_instance->getDimensions();

                    T* centroids = std::get<0>(results);
                    emscripten::val jsCentroids = emscripten::val::array();
                    long int k = m_instance->getK();
                    for (size_t i = 0; i < k; ++i) {
                        jsCentroids.call<void>("push", wasm::utility::contiguousVecToArray<T>(&centroids[i * dimensions], dimensions));
                    }
                    free(centroids);

                    long int* clusters = std::get<1>(results);
                    emscripten::val jsClusters = wasm::utility::contiguousVecToArray<long int>(clusters, dataLength / dimensions);
                    free(clusters);
                    
                    return KResult{jsCentroids, jsClusters};
                }
        };

        template <typename T>
        class KMedian {
            static const inline std::unordered_map<std::string, T (* )(T*, T*, long int)> distance_funcs = {
                { "euclidean", euclidean<T> }
            };

            public:
                KMedian(const long int k, const long int max_iterations, const T tolerance, const std::string distanceFunc) {
                    if (distanceFunc != "euclidean"){
                        throw std::invalid_argument(distanceFunc + " is not a valid distance metric");
                    }
                    m_distance_func = distanceFunc;
                    auto m_distance = distance::euclidean<T>;
                    m_instance = new clustering::KMedian<T>(k, max_iterations, tolerance, m_distance);
                }

                KResult predict(emscripten::val jsData) {
                    std::vector<std::vector<T>> data = wasm::utility::array2DToVec<T>(jsData);
                    auto results = this->m_instance->predict(data);

                    // convert data to Javascript
                    auto centroids = std::get<0>(results);
                    emscripten::val jsCentroids = emscripten::val::array();
                    for (auto & centroid : centroids) {
                        jsCentroids.call<void>("push", wasm::utility::vecToArray<T>(centroid));
                    }

                    auto clusters = std::get<1>(results);
                    emscripten::val jsClusters = wasm::utility::vecToArray<long int>(clusters);
                    return KResult{ jsCentroids, jsClusters};
                }

                void setK(const long int k) {
                    this->m_instance->setK(k);
                }

                long int getK() {
                    return this->m_instance->getK();
                }

                void setMaxIterations(const long int maxIterations) {
                    this->m_instance->setMaxIterations(maxIterations);
                }

                long int getMaxIterations() {
                    return this->m_instance->getMaxIterations();
                }

                void setTolerance(const T tolerance) {
                    this->m_instance->setTolerance(tolerance);
                }

                T getTolerance() {
                    return this->m_instance->getTolerance();
                }

                std::string getDistanceFunc() {
                    return this->m_distance_func;
                }

            private:
                clustering::KMedian<T> * m_instance;
                std::string m_distance_func;
        };
        
        template <typename T>
        class KMode {
            static const inline std::unordered_map<std::string, T (* )(T*, T*, long int)> distance_funcs = {
                { "euclidean", euclidean<T> }
            };

            public:
                KMode(const long int k, const long int max_iterations, const T tolerance, const std::string distanceFunc) {
                    if (distanceFunc != "euclidean"){
                        throw std::invalid_argument(distanceFunc + " is not a valid distance metric");
                    }
                    m_distance_func = distanceFunc;
                    auto m_distance = distance::euclidean<T>;
                    m_instance = new clustering::KMode<T>(k, max_iterations, tolerance, m_distance);
                }

                KResult predict(emscripten::val jsData) {
                    std::vector<std::vector<T>> data = wasm::utility::array2DToVec<T>(jsData);
                    auto results = this->m_instance->predict(data);

                    // convert data to Javascript
                    auto centroids = std::get<0>(results);
                    emscripten::val jsCentroids = emscripten::val::array();
                    for (auto & centroid : centroids) {
                        jsCentroids.call<void>("push", wasm::utility::vecToArray<T>(centroid));
                    }

                    auto clusters = std::get<1>(results);
                    emscripten::val jsClusters = wasm::utility::vecToArray<long int>(clusters);
                    return KResult{ jsCentroids, jsClusters};
                }

                void setK(const long int k) {
                    this->m_instance->setK(k);
                }

                long int getK() {
                    return this->m_instance->getK();
                }

                void setMaxIterations(const long int maxIterations) {
                    this->m_instance->setMaxIterations(maxIterations);
                }

                long int getMaxIterations() {
                    return this->m_instance->getMaxIterations();
                }

                void setTolerance(const T tolerance) {
                    this->m_instance->setTolerance(tolerance);
                }

                T getTolerance() {
                    return this->m_instance->getTolerance();
                }

                std::string getDistanceFunc() {
                    return this->m_distance_func;
                }

            private:
                clustering::KMode<T> * m_instance;
                std::string m_distance_func;
        };

    }
}

#endif /* WASM_KMEANS_H */