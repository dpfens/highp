#include <emscripten/bind.h>
#include "kmeans.cpp"
#include "kdtree/kdtree.cpp"

using namespace emscripten;
using namespace clustering;

EMSCRIPTEN_BINDINGS(highp) {
    class_<KMeans<double>>("KMeans")
        .constructor<int, int, double, double (*)(std::vector<double>, std::vector<double>)>()
        .function("setK", &KMeans<double>::setK)
        .function("getK", &KMeans<double>::getK)
        .function("setMaxIterations", &KMeans<double>::setMaxIterations)
        .function("getMaxIterations", &KMeans<double>::getMaxIterations)
        .function("setTolerance", &KMeans<double>::setTolerance)
        .function("getTolerance", &KMeans<double>::getTolerance)
        .function("predict", &KMeans<double>::predict);
    
    class_<KMedian<double>>("KMedian")
        .constructor<int, int, double, double (*)(std::vector<double>, std::vector<double>)>()
        .function("setK", &KMedian<double>::setK)
        .function("getK", &KMedian<double>::getK)
        .function("setMaxIterations", &KMedian<double>::setMaxIterations)
        .function("getMaxIterations", &KMedian<double>::getMaxIterations)
        .function("setTolerance", &KMedian<double>::setTolerance)
        .function("getTolerance", &KMedian<double>::getTolerance)
        .function("predict", &KMedian<double>::predict);

    class_<KMode<double>>("KMode")
        .constructor<int, int, double, double (*)(std::vector<double>, std::vector<double>)>()
        .function("setK", &KMode<double>::setK)
        .function("getK", &KMode<double>::getK)
        .function("setMaxIterations", &KMode<double>::setMaxIterations)
        .function("getMaxIterations", &KMode<double>::getMaxIterations)
        .function("setTolerance", &KMode<double>::setTolerance)
        .function("getTolerance", &KMode<double>::getTolerance)
        .function("predict", &KMode<double>::predict);
}

int main() {

}