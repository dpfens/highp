#include <vector>
#include <math.h>

namespace distance {
    template <typename T>
    T euclidean_distance(std::vector<T> point1, std::vector<T> point2) {
        long double distance = 0.0;
        if (point1.size() != point2.size()){
            return -1;
        }
        for (std::size_t i = 0; i < point1.size(); i++){
            distance += pow(point2[i] - point1[i], 2);
        }
        return sqrt(distance);
    }
}
