%module similarity
%{
/* Put header files here or function declarations like below */
#include "../src/cpp/similarity.hpp"
%}
%include "std_set.i"

namespace std {
    %template(setli) std::set<long int>;
}

%include "../src/cpp/similarity.hpp"

%pythoncallback;
%template(jaccard) similarity::jaccard<long int>;
%pythoncallback;
%template(sorensen_dice) similarity::sorensen_dice<long int>;
%pythoncallback;
%template(overlap) similarity::overlap<long int>;

%pythoncallback;
%template(hyh) similarity::fuzzy::hwang_yang_hung<int>;

%pythoncallback;
%template(ls) similarity::fuzzy::liang_shi<int>;

%pythoncallback;
%template(dc) similarity::fuzzy::dengfeng_chuntian<int>;
