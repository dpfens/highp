#include <emscripten/bind.h>
#include "kmeans.cpp"
#include "dbscan.cpp"
#include "fuzzy_pack.cpp"

using namespace emscripten;
using namespace clustering;
using namespace density;

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
    
    // Binding for DBSCAN class
    class_<DBSCAN<double>>("DBSCAN")
        .constructor<double, long int, double (*)(std::vector<double>, std::vector<double>)>()
        .function("predict", &DBSCAN<double>::predict);

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
}

int main() {

}