#include <emscripten/val.h>
#include "../kmeans.cpp"
#include "../distance.cpp"

#include "utility.hpp"


namespace Wasm {

    struct KResult {
        emscripten::val centroids;
        emscripten::val clusters;
    };

    template <typename T>
    class KMeans {
        public:
            KMeans(const long int k, const long int max_iterations, const double tolerance, const std::string distanceFunc) {
                if (distanceFunc != "euclidean"){
                    throw std::invalid_argument(distanceFunc + " is not a valid distance metric");
                }
                m_distance_func = distanceFunc;
                auto m_distance = distance::euclidean<T>;
                m_instance = new clustering::KMeans<T>(k, max_iterations, tolerance, m_distance);
            }

            KResult predict(std::vector<std::vector<T> > &data) {
                auto results = this->m_instance->predict(data);

                // convert data to Javascript
                auto centroids = std::get<0>(results);
                emscripten::val jsCentroids = emscripten::val::array();
                for (auto & centroid : centroids) {
                    jsCentroids.call<void>("push", Wasm::Utility::vecToArray<T>(centroid));
                }

                auto clusters = std::get<1>(results);
                emscripten::val jsClusters = Wasm::Utility::vecToArray<long int>(clusters);
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
            clustering::KMeans<T> * m_instance;
            std::string m_distance_func;
    };

}