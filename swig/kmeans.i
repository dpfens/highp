%module kmeans
%{
#include "../src/cpp/kmeans.hpp"
%}

%include "std_vector.i"

namespace std {
    %template(vectori) std::vector<int>;
    %template(vectord) std::vector<double>;
    %template(VecVecdouble) std::vector< std::vector<double> >;
}

%include "../src/cpp/kmeans.hpp"

%template(StrictKMeans) clustering::KMeans<double>;
%template(StrictKMedian) clustering::KMedian<double>;
%template(StrictKMode) clustering::KMode<double>;
