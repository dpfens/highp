#include <vector>
#include <tuple>
#include <emscripten/val.h>
#include <emscripten/bind.h>
#include "kmeans.cpp"
#include "dbscan.cpp"
#include "fuzzy_pack.cpp"
#include "../fuzzy.cpp"

using namespace emscripten;
using namespace clustering;
using namespace density;


std::vector<int> returnVectorData () {
  std::vector<int> v(10, 1);
  return v;
}


EMSCRIPTEN_BINDINGS(highp) {
    register_vector<int>("VectorInt");
    register_vector<long int>("VectorLongInt");
    register_vector<double>("VectorDouble");
    register_vector<std::vector<double>>("VectorMatrixDouble");
    register_vector<std::string>("VectorString");

    register_map<long int, double>("FuzzyClusterMap");

    function("returnVectorData", &returnVectorData);
    function("JsArrayToVectorDouble", &emscripten::vecFromJSArray<double>);
    function("JsArrayToVectorInt", &emscripten::vecFromJSArray<int>);
    function("JsArrayToVectorString", &emscripten::vecFromJSArray<std::string>);

    value_object<wasm::cluster::KResult>("KResult")
        .field("centroids", &wasm::cluster::KResult::centroids)
        .field("clusters", &wasm::cluster::KResult::clusters);

    class_<wasm::cluster::KMeans<double>>("KMeans")
        .constructor<int, int, double, std::string>()
        .function("setK", &wasm::cluster::KMeans<double>::setK)
        .function("getK", &wasm::cluster::KMeans<double>::getK)
        .function("setMaxIterations", &wasm::cluster::KMeans<double>::setMaxIterations)
        .function("getMaxIterations", &wasm::cluster::KMeans<double>::getMaxIterations)
        .function("setTolerance", &wasm::cluster::KMeans<double>::setTolerance)
        .function("getTolerance", &wasm::cluster::KMeans<double>::getTolerance)
        .function("getDistance", &wasm::cluster::KMeans<double>::getDistanceFunc)
        .function("predict", &wasm::cluster::KMeans<double>::predict);
    
    class_<wasm::cluster::KMedian<double>>("KMedian")
        .constructor<int, int, double, std::string>()
        .function("setK", &wasm::cluster::KMedian<double>::setK)
        .function("getK", &wasm::cluster::KMedian<double>::getK)
        .function("setMaxIterations", &wasm::cluster::KMedian<double>::setMaxIterations)
        .function("getMaxIterations", &wasm::cluster::KMedian<double>::getMaxIterations)
        .function("setTolerance", &wasm::cluster::KMedian<double>::setTolerance)
        .function("getTolerance", &wasm::cluster::KMedian<double>::getTolerance)
        .function("getDistance", &wasm::cluster::KMedian<double>::getDistanceFunc)
        .function("predict", &wasm::cluster::KMedian<double>::predict);

    class_<wasm::cluster::KMode<double>>("KMode")
        .constructor<int, int, double, std::string>()
        .function("setK", &wasm::cluster::KMode<double>::setK)
        .function("getK", &wasm::cluster::KMode<double>::getK)
        .function("setMaxIterations", &wasm::cluster::KMode<double>::setMaxIterations)
        .function("getMaxIterations", &wasm::cluster::KMode<double>::getMaxIterations)
        .function("setTolerance", &wasm::cluster::KMode<double>::setTolerance)
        .function("getTolerance", &wasm::cluster::KMode<double>::getTolerance)
        .function("getDistance", &wasm::cluster::KMode<double>::getDistanceFunc)
        .function("predict", &wasm::cluster::KMode<double>::predict);
    
    class_<wasm::cluster::DBSCAN<double>>("DBSCAN")
        .constructor<double, long int, std::string>()
        .function("setEpsilon", &wasm::cluster::DBSCAN<double>::setEpsilon)
        .function("getEpsilon", &wasm::cluster::DBSCAN<double>::getEpsilon)
        .function("setMinPoints", &wasm::cluster::DBSCAN<double>::setMinPoints)
        .function("getMinPoints", &wasm::cluster::DBSCAN<double>::getMinPoints)
        .function("getDistance", &wasm::cluster::DBSCAN<double>::getDistanceFunc)
        .function("predict", &wasm::cluster::DBSCAN<double>::predict);
    
    // Binding for DBPack class
    class_<wasm::cluster::DBPack<double>>("DBPack")
        .constructor<double, long int>()
        .function("setEpsilon", &wasm::cluster::DBPack<double>::setEpsilon)
        .function("getEpsilon", &wasm::cluster::DBPack<double>::getEpsilon)
        .function("setMinPoints", &wasm::cluster::DBPack<double>::setMinPoints)
        .function("getMinPoints", &wasm::cluster::DBPack<double>::getMinPoints)
        .function("predict", &DBPack<double>::predict);
    
    // Binding for CoreDBPack class
    class_<wasm::cluster::fuzzy::CoreDBPack<double, long int>>("CoreDBPack")
        .constructor<double, long int, long int>()
        .function("setMinEpsilon", &wasm::cluster::fuzzy::CoreDBPack<double, long int>::setMinEpsilon)
        .function("getMinEpsilon", &wasm::cluster::fuzzy::CoreDBPack<double, long int>::getMinEpsilon)
        .function("setMinPoints", &wasm::cluster::fuzzy::CoreDBPack<double, long int>::setMinPoints)
        .function("getMinPoints", &wasm::cluster::fuzzy::CoreDBPack<double, long int>::getMinPoints)
        .function("setMaxPoints", &wasm::cluster::fuzzy::CoreDBPack<double, long int>::setMaxPoints)
        .function("getMaxPoints", &wasm::cluster::fuzzy::CoreDBPack<double, long int>::getMaxPoints)
        .function("predict", &wasm::cluster::fuzzy::CoreDBPack<double, long int>::predict);

    // Binding for BorderDBPack class
    class_<wasm::cluster::fuzzy::BorderDBPack<double, long int>>("BorderDBPack")
        .constructor<double, double, long int>()
        .function("setMinEpsilon", &wasm::cluster::fuzzy::BorderDBPack<double, long int>::setMinEpsilon)
        .function("getMinEpsilon", &wasm::cluster::fuzzy::BorderDBPack<double, long int>::getMinEpsilon)
        .function("setMaxEpsilon", &wasm::cluster::fuzzy::BorderDBPack<double, long int>::setMaxEpsilon)
        .function("getMaxEpsilon", &wasm::cluster::fuzzy::BorderDBPack<double, long int>::getMaxEpsilon)
        .function("setMinPoints", &wasm::cluster::fuzzy::BorderDBPack<double, long int>::setMinPoints)
        .function("getMinPoints", &wasm::cluster::fuzzy::BorderDBPack<double, long int>::getMinPoints)
        .function("predict", &wasm::cluster::fuzzy::BorderDBPack<double, long int>::predict);

    function("sad", &distance::sad<double>);
    function("euclidean", &distance::euclidean<double>);
}

int main() {

}