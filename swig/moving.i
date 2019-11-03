
 %module moving
 %{
 /* Put header files here or function declarations like below */
 using namespace std;
 #include "../src/cpp/moving.hpp"
 %}

 %include "std_vector.i"

 namespace std {
     %template(vectori) std::vector<int>;
     %template(vectord) std::vector<double>;
     %template(VecVecdouble) std::vector< std::vector<double> >;
     %template(VecVecVecdouble) std::vector<std::vector< std::vector<double> > >;
 }
 %include "../src/cpp/moving.hpp"

 %template(NormalMovingDBSCAN) density::moving::MovingDBSCAN<double>;
