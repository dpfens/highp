 %module fuzzy
 %{
 /* Put header files here or function declarations like below */
 #include "../src/cpp/fuzzy.hpp"
 %}

 %include "std_vector.i"
 %include "std_map.i"

 namespace std {
     %template(vectori) std::vector<int>;
     %template(vectord) std::vector<double>;
     %template(VecVecdouble) std::vector< std::vector<double> >;
     %template(MapID) std::map<int, double>;
     %template(VectorMap) std::vector<std::map<int, double> >;
 }

%include "../src/cpp/fuzzy.hpp"

 %template(FuzzyBaseDBSCAN) density::fuzzy::BaseDBSCAN<double>;
 %template(FuzzyCoreDBSCAN) density::fuzzy::CoreDBSCAN<double>;
 %template(FuzzyBorderDBSCAN) density::fuzzy::BorderDBSCAN<double>;
 %template(FuzzyDBSCAN) density::fuzzy::DBSCAN<double>;
