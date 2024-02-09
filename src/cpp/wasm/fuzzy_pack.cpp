#ifndef WASM_FUZZY_PACK_H
#define WASM_FUZZY_PACK_H

#include "../fuzzy_pack.cpp"

#include <emscripten/val.h>
#include "../distance.cpp"

#include "utility.hpp"


namespace wasm {

    namespace cluster {

        namespace fuzzy {

            template <class T1, class T2>
            class CoreDBPack {
                public:
                    CoreDBPack(T1 min_epsilon, T2 min_points, T2 max_points) {
                        m_instance = new density::fuzzy::CoreDBPack<T1, T2>(min_epsilon, min_points, max_points);
                    }

                    emscripten::val predict(emscripten::val jsData) {
                        std::vector<T1> data = wasm::utility::arrayToVec<T1>(jsData);
                        auto fuzzyClusterSet = this->m_instance->predict(data);

                        // convert maps to JS Objects
                        emscripten::val jsClusters = emscripten::val::array();
                        for (auto & fuzzyClusters : fuzzyClusterSet) {
                            jsClusters.call<void>("push", wasm::utility::mapToObject<T2, T1>(fuzzyClusters));
                        }

                        return jsClusters;
                    }

                    void setMinEpsilon(const T1 value) {
                        this->m_instance->setMinEpsilon(value);
                    }

                    T1 getMinEpsilon() {
                        return this->m_instance->getMinEpsilon();
                    }

                    void setMinPoints(const T2 value) {
                        this->m_instance->setMinPoints(value);
                    }

                    T2 getMinPoints() {
                        return this->m_instance->getMinPoints();
                    }

                    void setMaxPoints (const T2 maxPoints) {
                        this->m_instance->setMaxPoints(maxPoints);
                    }

                    T2 getMaxPoints () {
                        return this->m_instance->getMaxPoints();
                    }

                private:
                    density::fuzzy::CoreDBPack<T1, T2> * m_instance;
            };

            template <class T1, class T2>
            class BorderDBPack {
                public:
                    BorderDBPack(T1 min_epsilon, T1 max_epsilon, T2 min_points) {
                        m_instance = new density::fuzzy::BorderDBPack<T1, T2>(min_epsilon, max_epsilon, min_points);
                    }

                    emscripten::val predict(emscripten::val jsData) {
                        std::vector<T1> data = wasm::utility::arrayToVec<T1>(jsData);
                        auto fuzzyClusterSet = this->m_instance->predict(data);

                        // convert maps to JS Objects
                        emscripten::val jsClusters = emscripten::val::array();
                        for (auto & fuzzyClusters : fuzzyClusterSet) {
                            jsClusters.call<void>("push", wasm::utility::mapToObject<T2, T1>(fuzzyClusters));
                        }

                        return jsClusters;
                    }

                    void setMinEpsilon(const T1 value) {
                        this->m_instance->setMinEpsilon(value);
                    }

                    T1 getMinEpsilon() {
                        return this->m_instance->getMinEpsilon();
                    }

                    void setMaxEpsilon(const T1 value) {
                        this->m_instance->setMaxEpsilon(value);
                    }

                    T1 getMaxEpsilon() {
                        return this->m_instance->getMaxEpsilon();
                    }

                    void setMinPoints(const T2 value) {
                        this->m_instance->setMinPoints(value);
                    }

                    T2 getMinPoints() {
                        return this->m_instance->getMinPoints();
                    }

                private:
                    density::fuzzy::BorderDBPack<T1, T2> * m_instance;
            };
        }
    }
}

#endif /* WASM_FUZZY_PACK_H */