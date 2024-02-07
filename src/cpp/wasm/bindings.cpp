#include <vector>
#include <tuple>
#include <emscripten/val.h>
#include <emscripten/bind.h>
#include "kmeans.cpp"
#include "dbscan.cpp"
#include "../fuzzy_pack.cpp"
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

    function("returnVectorData", &returnVectorData);
    function("JsArrayToVectorDouble", &emscripten::vecFromJSArray<double>);
    function("JsArrayToVectorInt", &emscripten::vecFromJSArray<int>);
    function("JsArrayToVectorString", &emscripten::vecFromJSArray<std::string>);

    value_object<Wasm::KResult>("KResult")
        .field("centroids", &Wasm::KResult::centroids)
        .field("clusters", &Wasm::KResult::clusters);

    class_<Wasm::KMeans<double>>("KMeans")
        .constructor<int, int, double, std::string>()
        .function("setK", &Wasm::KMeans<double>::setK)
        .function("getK", &Wasm::KMeans<double>::getK)
        .function("setMaxIterations", &Wasm::KMeans<double>::setMaxIterations)
        .function("getMaxIterations", &Wasm::KMeans<double>::getMaxIterations)
        .function("setTolerance", &Wasm::KMeans<double>::setTolerance)
        .function("getTolerance", &Wasm::KMeans<double>::getTolerance)
        .function("getDistance", &Wasm::KMeans<double>::getDistanceFunc)
        .function("predict", &Wasm::KMeans<double>::predict);
    
    // Binding for DBPack class
    class_<DBPack<double>>("DBPack")
        .constructor<double>()
        .function("predict", &DBPack<double>::predict);

    // Binding for DBPack2 class
    class_<DBPack2<double>>("DBPack2")
        .constructor<double, unsigned long int>()
        .function("predict", &DBPack2<double>::predict);
    
    // Binding for CoreDBPack class
    class_<fuzzy::CoreDBPack<double, long int>>("CoreDBPack")
        .constructor<double, long int, long int>()
        .function("predict", &fuzzy::CoreDBPack<double, long int>::predict);

    // Binding for BorderDBPack class
    class_<fuzzy::BorderDBPack<double, long int>>("BorderDBPack")
        .constructor<double, double, long int>()
        .function("predict", &fuzzy::BorderDBPack<double, long int>::predict);

    function("sad", &distance::sad<double>);
    function("euclidean", &distance::euclidean<double>);
}

int main() {

}