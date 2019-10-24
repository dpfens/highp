%module distance
%{
/* Put header files here or function declarations like below */
#include "../src/cpp/distance.hpp"
%}
%include "std_vector.i"

namespace std {
    %template(vectord) std::vector<double>;
}

%include "../src/cpp/distance.hpp"

%pythoncallback;
%template(euclidean) distance::euclidean_distance<double>;
