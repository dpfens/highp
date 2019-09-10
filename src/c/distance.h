#include <math.h>
#define M_PI (4.0 * atan(1.0))

long double euclidean_distance(long double * point1, long double * point2, unsigned long long int dimension_count) {
  long double distance = 0.0;
  size_t i = 0;
  for(i = 0; i < dimension_count; ++i) {
    distance += pow(point1[i] - point2[i], 2);
  }
  return sqrt(distance);
}

long double to_radians(long double degrees) {
    return degrees * (M_PI / 180.0);
}

long double great_circle_distance(long double * point1, long double * point2, unsigned long long int dimension_count) {
    long double latitude_1 = to_radians(point1[0]);
    long double longitude_1 = to_radians(point1[0]);

    long double latitude_2 = to_radians(point2[0]);
    long double longitude_2 = to_radians(point2[0]);

    long double latitude_distance = latitude_2 - latitude_1;
    long double longitude_distance = longitude_2 - longitude_1;

    long double a = pow(sin(latitude_distance / 2), 2) + cos(latitude_1) * cos(latitude_2) * pow(sin(longitude_distance / 2), 2);
    long double c = 2 * asin(sqrt(a));
    return 6371 * c;
}
