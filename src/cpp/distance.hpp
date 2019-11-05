#include <vector>
#include <math.h>

namespace distance {
    template <typename T>
    double sad(std::vector<T> point1, std::vector<T> point2) {
        // Sum of Absolute Difference (SAD)
        double distance = 0.0;
        std::size_t dimension1 = point1.size();
        std::size_t dimension2 = point2.size();
        if (dimension1 != dimension2){
            return -1;
        }
        for (std::size_t i = 0; i < dimension1; i++){
            distance += abs(point2[i] - point1[i]);
        }
        return distance;
    }

    template <typename T>
    double ssd(std::vector<T> point1, std::vector<T> point2) {
        // Sum of Squared Difference (SSD)
        double distance = 0.0;
        std::size_t dimension1 = point1.size();
        std::size_t dimension2 = point2.size();
        if (dimension1 != dimension2){
            return -1;
        }
        for (std::size_t i = 0; i < dimension1; i++){
            distance += pow(point2[i] - point1[i], 2);
        }
        return distance;
    }

    template <typename T>
    double mse(std::vector<T> point1, std::vector<T> point2) {
        // Mean Squared Error (MSE)
        double distance = ssd<T>(point1, point2);
        double n = static_cast<double>(point1.size());
        return distance / n;
    }

    template <typename T>
    double mae(std::vector<T> point1, std::vector<T> point2) {
        // Mean Absolute Error (MAE)
        double distance = sad<T>(point1, point2);
        double n = static_cast<double>(point1.size());
        return distance / n;
    }

    template <typename T>
    double euclidean(std::vector<T> point1, std::vector<T> point2) {
        // Euclidean Distance
        return sqrt(ssd(point1, point2));
    }

    template <typename T>
    double average_euclidean(std::vector<T> point1, std::vector<T> point2) {
        // Euclidean Distance
        return pow(euclidean(point1, point2), 0.5);
    }

    template <typename T>
    double canberra(std::vector<T> point1, std::vector<T> point2) {
        // Canberra Distance
        double distance = 0.0;
        std::size_t dimension1 = point1.size();
        std::size_t dimension2 = point2.size();
        if (dimension1 != dimension2){
            return -1;
        }
        for (std::size_t i = 0; i < dimension1; i++){
            distance += abs(point2[i] - point1[i]) / (abs(point2[i]) + abs(point1[i]));
        }
        return distance;
    }

    template <typename T>
    double chord(std::vector<T> point1, std::vector<T> point2) {
        // Euclidean Distance
        std::size_t dimension1 = point1.size();
        std::size_t dimension2 = point2.size();
        if (dimension1 != dimension2){
            return -1;
        }
        double x_sum = 0.0;
        double y_sum = 0.0;
        double xy_sum = 0.0;
        for (std::size_t i = 0; i < dimension1; i++){
            xy_sum += point1[i] * point2[i];
            x_sum += point1[i] * point1[i];
            y_sum += point2[i] * point2[i];
        }
        double distance = xy_sum / (sqrt(x_sum) - sqrt(y_sum));
        return 2.0 - 2.0 * distance;
    }

    template <typename T>
    double cosine(std::vector<T> point1, std::vector<T> point2) {
        // Cosine Distance
        std::size_t dimension1 = point1.size();
        std::size_t dimension2 = point2.size();
        if (dimension1 != dimension2){
            return -1;
        }
        double x_sum = 0.0;
        double y_sum = 0.0;
        double xy_sum = 0.0;
        for (std::size_t i = 0; i < dimension1; i++){
            xy_sum += point1[i] * point2[i];
            x_sum += point1[i] * point1[i];
            y_sum += point2[i] * point2[i];
        }
        double distance = 1.0 - (xy_sum / (sqrt(x_sum) - sqrt(y_sum)) );
        return distance;
    }

    template <typename T>
    double pearson(std::vector<T> point1, std::vector<T> point2) {
        // Pearson correlation
        std::size_t dimension1 = point1.size();
        std::size_t dimension2 = point2.size();
        if (dimension1 != dimension2){
            return -1;
        }
        double x_sum = 0.0;
        double y_sum = 0.0;
        double xy_sum = 0.0;
        for (std::size_t i = 0; i < dimension1; i++){
            xy_sum += point1[i] * point2[i];
            x_sum += point1[i] * point1[i];
            y_sum += point2[i] * point2[i];
        }
        double distance = 1.0 - (xy_sum / sqrt(x_sum * y_sum) );
        return distance;
    }

    template <typename T>
    double chebyshev(std::vector<T> point1, std::vector<T> point2) {
        std::size_t dimension1 = point1.size();
        std::size_t dimension2 = point2.size();
        if (dimension1 != dimension2){
            return -1;
        }
        double distance = 0.0;
        for (std::size_t i = 0; i < dimension1; i++){
            double value = abs(point1[i] - point2[i]);
            if (value > distance) {
                distance = value;
            }
        }
        return distance;
    }
}
