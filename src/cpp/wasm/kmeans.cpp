#ifndef WASM_KMEANS_H
#define WASM_KMEANS_H

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "../kmeans.cpp"
#include "../distance.hpp"

#include "utility.hpp"

namespace wasm {

    namespace cluster {

        struct KResult {
            emscripten::val centroids;
            emscripten::val clusters;
        };

        template <typename T>
        double ssd(T* point1, T* point2, long int dimensions) {
            // Sum of Squared Difference (SSD)
            T distance = 0.0;
            for (std::size_t i = 0; i < dimensions; i++){
                distance += pow(point2[i] - point1[i], 2);
            }
            return distance;
        }

        template <typename T>
        double euclidean(T* point1, T* point2, long int dimensions) {
            // Euclidean Distance
            return sqrt(ssd<T>(point1, point2, dimensions));
        }

        template <typename T>
        class KMeans {
            public:
                KMeans(const long int k, const long int max_iterations, const double tolerance, long int dimensions, const std::string distanceFunc) {
                    if (distanceFunc != "euclidean"){
                        throw std::invalid_argument(distanceFunc + " is not a valid distance metric");
                    }
                    m_distance_func = distanceFunc;
                    m_instance = new clustering::KMeansContiguous<T>(k, max_iterations, tolerance, dimensions, euclidean<T>);
                }

                KResult predict(emscripten::val jsData) {
                    // convert TypedArray to a T* pointer
                    unsigned int jsDataLength= jsData["length"].as<long int>();
                    emscripten::val buffer = jsData["buffer"]; 
                    std::vector<T> byte_data = emscripten::convertJSArrayToNumberVector<T>(jsData);
                    T* data = reinterpret_cast<T*>(&byte_data[0]); 

                    auto results = this->m_instance->predict(data, jsDataLength);
                    free(data);

                    // convert data to Javascript
                    long int dimensions = m_instance->getDimensions();

                    T* centroids = std::get<0>(results);
                    emscripten::val jsCentroids = emscripten::val::array();
                    long int k = m_instance->getK();
                    for (size_t i = 0; i < k; ++i) {
                        jsCentroids.call<void>("push", wasm::utility::contiguousVecToArray<T>(&centroids[i * dimensions], dimensions));
                    }
                    free(centroids);

                    long int * clusters = std::get<1>(results);
                    emscripten::val jsClusters = wasm::utility::contiguousVecToArray<long int>(clusters, jsDataLength / dimensions);
                    free(clusters);
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

                void setTolerance(const double tolerance) {
                    this->m_instance->setTolerance(tolerance);
                }

                double getTolerance() {
                    return this->m_instance->getTolerance();
                }

                std::string getDistanceFunc() {
                    return this->m_distance_func;
                }

            private:
                clustering::KMeansContiguous<T> * m_instance;
                std::string m_distance_func;
        };

        template <typename T>
        class KMedian {
            public:
                KMedian(const long int k, const long int max_iterations, const double tolerance, const std::string distanceFunc) {
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

                void setTolerance(const double tolerance) {
                    this->m_instance->setTolerance(tolerance);
                }

                double getTolerance() {
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
            public:
                KMode(const long int k, const long int max_iterations, const double tolerance, const std::string distanceFunc) {
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

                void setTolerance(const double tolerance) {
                    this->m_instance->setTolerance(tolerance);
                }

                double getTolerance() {
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