#include <set>

namespace similarity {

    template <typename T>
    double jaccard(std::set<T> point1, std::set<T> point2) {
        std::set<T> intersection;
        set_intersection(point1.begin(), point1.end(), point2.begin(), point2.end(),
                  std::inserter(intersection, intersection.begin()));
        double intersection_size = static_cast<double>(intersection.size());
        std::set<T> un;
        set_union(point1.begin(), point1.end(), point2.begin(), point2.end(),
                  std::inserter(un, un.begin()));
        double union_size = static_cast<double>(un.size());
        return intersection_size / union_size;
    }

    template <typename T>
    double sorensen_dice(std::set<T> point1, std::set<T> point2) {
        // overlap coefficient
        std::set<T> intersection;
        set_intersection(point1.begin(), point1.end(), point2.begin(), point2.end(),
                  std::inserter(intersection, intersection.begin()));
        double intersection_size = static_cast<double>(intersection.size());
        std::size_t point1_size = point1.size();
        std::size_t point2_size = point2.size();
        std::size_t denominator = point1_size + point2_size;
        return (2 * intersection_size) / static_cast<double>(denominator);
    }

    template <typename T>
    double overlap(std::set<T> point1, std::set<T> point2) {
        // overlap coefficient
        std::set<T> intersection;
        set_intersection(point1.begin(), point1.end(), point2.begin(), point2.end(),
                  std::inserter(intersection, intersection.begin()));
        double intersection_size = static_cast<double>(intersection.size());
        std::size_t point1_size = point1.size();
        std::size_t point2_size = point2.size();
        std::size_t denominator = point1_size;
        if (point2_size < point1_size) {
            denominator = point2_size;
        }
        return intersection_size / static_cast<double>(denominator);
    }
}
