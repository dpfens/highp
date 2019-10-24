/* example.i */

 %module dbscan
 %{
 /* Put header files here or function declarations like below */
 using namespace std;
 #include "../src/cpp/dbscan.hpp"
 %}

 %include "std_vector.i"

 namespace std {
     %template(vectori) std::vector<int>;
     %template(vectord) std::vector<double>;
     %template(VecVecdouble) std::vector< std::vector<double> >;
 }
 %include "../src/cpp/dbscan.hpp"

 %template(NormalDBSCAN) density::DBSCAN<double>;
