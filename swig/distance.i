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
%template(sad) distance::sad<double>;
%pythoncallback;
%template(ssd) distance::ssd<double>;
%pythoncallback;
%template(mse) distance::mse<double>;
%pythoncallback;
%template(mae) distance::mae<double>;
%pythoncallback;
%template(canberra) distance::canberra<double>;
%pythoncallback;
%template(chord) distance::chord<double>;
%pythoncallback;
%template(cosine) distance::cosine<double>;
%pythoncallback;
%template(pearson) distance::pearson<double>;
%pythoncallback;
%template(average_euclidean) distance::average_euclidean<double>;
%pythoncallback;
%template(euclidean) distance::euclidean<double>;
%pythoncallback;
%template(chebyshev) distance::chebyshev<double>;
