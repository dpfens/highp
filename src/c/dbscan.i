%module cluster
%{
/* Put header files here or function declarations like below */
#include <stdbool.h>
extern long long int * dbscan(long double ** data, unsigned long long int sample_count, unsigned long long int dimension_count, long double eps, long long int min_points, long double (*distance_func)(long double *, long double *, unsigned long long int), bool verbose);
%}

extern long long int * dbscan(long double ** data, unsigned long long int sample_count, unsigned long long int dimension_count, long double eps, long long int min_points, long double (*distance_func)(long double *, long double *, unsigned long long int), bool verbose);
