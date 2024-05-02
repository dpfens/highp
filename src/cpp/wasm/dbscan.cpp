#ifndef WASM_DBSCAN_H
#define WASM_DBSCAN_H

#include "../dbscan.cpp"

#include <emscripten/val.h>
#include "../distance.cpp"

#include "utility.hpp"


namespace wasm {

    namespace cluster {

        template <typename T>
        class DBSCAN {
            public:
                DBSCAN(const T epsilon, const long int min_points, const std::string distanceFunc) {
                    if (distanceFunc != "euclidean"){
                        throw std::invalid_argument(distanceFunc + " is not a valid distance metric");
                    }
                    m_distance_func = distanceFunc;
                    auto m_distance = distance::euclidean<T>;
                    m_instance = new density::DBSCAN<T>(epsilon, min_points, m_distance);
                }

                emscripten::val predict(emscripten::val jsData) {
                    std::vector<std::vector<T>> data = wasm::utility::array2DToVec<T>(jsData);
                    auto clusters = this->m_instance->predict(data);
                    return wasm::utility::vecToArray<int>(clusters);
                }

                void setEpsilon(const T value) {
                    this->m_instance->setEpsilon(value);
                }

                long int getEpsilon() {
                    return this->m_instance->getEpsilon();
                }

                void setMinPoints(const long int value) {
                    this->m_instance->setMinPoints(value);
                }

                long int getMinPoints() {
                    return this->m_instance->getMinPoints();
                }

                std::string getDistanceFunc() {
                    return this->m_distance_func;
                }

            private:
                density::DBSCAN<T> * m_instance;
                std::string m_distance_func;
        };

        template <typename T>
        class DBPack {
            public:
                DBPack(const T epsilon, const long int min_points) {
                    m_instance = new density::DBPack<T>(epsilon, min_points);
                }

                emscripten::val predict(emscripten::val jsData) {
                    std::vector<std::vector<T>> data = wasm::utility::array2DToVec<T>(jsData);
                    auto clusters = this->m_instance->predict(data);
                    return wasm::utility::vecToArray<int>(clusters);
                }

                void setEpsilon(const T value) {
                    this->m_instance->setEpsilon(value);
                }

                long int getEpsilon() {
                    return this->m_instance->getEpsilon();
                }

                void setMinPoints(const long int value) {
                    this->m_instance->setMinPoints(value);
                }

                long int getMinPoints() {
                    return this->m_instance->getMinPoints();
                }


            private:
                density::DBPack<T> * m_instance;
        };
    }
}

#endif /* WASM_DBSCAN_H */