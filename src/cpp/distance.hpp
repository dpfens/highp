#ifndef DISTANCE_H
#define DISTANCE_H

#include <vector>
#include <math.h>
#include <limits>

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

    template <class T, class T2>
    double hausdorff(std::vector<T> &point1, std::vector<T> &point2, double (* distance_func)(T, T)) {
        std::size_t point1_size = point1.size();
        std::size_t point2_size = point2.size();
        double max_double = std::numeric_limits<double>::max();
        double c_max = 0.0;

        std::size_t i = 0;
        #pragma omp parallel for private(i) shared(point1, point2)
        for (i = 0; i < point1_size; ++i) {
            double c_min = max_double;
            for (std::size_t j = 0; j < point2_size; ++j) {
                double distance = distance_func(point1[i], point2[j]);
                if (distance < c_min) {
                    c_min = distance;
                }
                if (c_min < c_max) {
                    break;
                }
            }
            #pragma omp critical
            if (c_min > c_max && max_double > c_min) {
                c_max = c_min;
            }
        }

        #pragma omp parallel for private(i) shared(point1, point2)
        for (i = 0; i < point2_size; ++i) {
            double c_min = max_double;
            for (std::size_t j = 0; j < point1_size; ++j) {
                double distance = distance_func(point2[i], point1[j]);
                if (distance < c_min) {
                    c_min = distance;
                }
                if (c_min < c_max) {
                    break;
                }
            }
            #pragma omp critical
            if (c_min > c_max && max_double > c_min) {
                c_max = c_min;
            }
        }
        return c_max;
    }


    namespace binary {
        template <typename T>
        double yuleqDistance(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a = 0.0,
            b = 0.0,
            c = 0.0,
            d = 0.0;
            for (std::size_t i = 0; i < d1; ++i) {
                if (obj1[i] && obj2[i]) {
                    ++a;
                } else if (obj1[i] && !obj2[i]) {
                    ++b;
                } else if (!obj1[i] && obj2[i]) {
                    ++c;
                } else {
                    ++d;
                }
            }
            double ad = a * d;
            double bc = b * c;
            double numerator = 2 * bc;
            double denominator = ad + bc;
            return numerator / denominator;
        }


        template <typename T>
        double hamming(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double b = 0.0,
            c = 0.0;
            for (std::size_t i = 0; i < d1; ++i) {
                if (obj1[i] && !obj2[i]) {
                    ++b;
                } else if (!obj1[i] && obj2[i]) {
                    ++c;
                }
            }
            return b + c;
        }


        template <typename T>
        double euclid(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double b = 0.0,
            c = 0.0;
            for (std::size_t i = 0; i < d1; ++i) {
                if (obj1[i] && !obj2[i]) {
                    ++b;
                } else if (!obj1[i] && obj2[i]) {
                    ++c;
                }
            }
            return sqrt(b + c);
        }


        template <typename T>
        double meanManhattan(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a = 0.0,
            b = 0.0,
            c = 0.0,
            d = 0.0;
            for (std::size_t i = 0; i < d1; ++i) {
                if (obj1[i] && obj2[i]) {
                    ++a;
                } else if (obj1[i] && !obj2[i]) {
                    ++b;
                } else if (!obj1[i] && obj2[i]) {
                    ++c;
                } else {
                    ++d;
                }
            }
            double n = a + b + c + d;
            return (b + c) / n;
        }


        template <typename T>
        double vari(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a = 0.0,
            b = 0.0,
            c = 0.0,
            d = 0.0;
            for (std::size_t i = 0; i < d1; ++i) {
                if (obj1[i] && obj2[i]) {
                    ++a;
                } else if (obj1[i] && !obj2[i]) {
                    ++b;
                } else if (!obj1[i] && obj2[i]) {
                    ++c;
                } else {
                    ++d;
                }
            }
            double n = a + b + c + d;
            return (b + c) / (4 * n);
        }


        template <typename T>
        double sizeDifference(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a = 0.0,
            b = 0.0,
            c = 0.0,
            d = 0.0;
            for (std::size_t i = 0; i < d1; ++i) {
                if (obj1[i] && obj2[i]) {
                    ++a;
                } else if (obj1[i] && !obj2[i]) {
                    ++b;
                } else if (!obj1[i] && obj2[i]) {
                    ++c;
                } else {
                    ++d;
                }
            }
            double n = a + b + c + d;
            return pow(b + c, 2) / pow(n, 2);
        }


        template <typename T>
        double shapeDifference(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a = 0.0,
            b = 0.0,
            c = 0.0,
            d = 0.0;
            for (std::size_t i = 0; i < d1; ++i) {
                if (obj1[i] && obj2[i]) {
                    ++a;
                } else if (obj1[i] && !obj2[i]) {
                    ++b;
                } else if (!obj1[i] && obj2[i]) {
                    ++c;
                } else {
                    ++d;
                }
            }
            double n = a + b + c + d;
            double numerator =  n * (b + c) - pow(b - c, 2);
            return numerator / pow(n, 2);
        }


        template <typename T>
        double patternDifference(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a = 0.0,
            b = 0.0,
            c = 0.0,
            d = 0.0;
            for (std::size_t i = 0; i < d1; ++i) {
                if (obj1[i] && obj2[i]) {
                    ++a;
                } else if (obj1[i] && !obj2[i]) {
                    ++b;
                } else if (!obj1[i] && obj2[i]) {
                    ++c;
                } else {
                    ++d;
                }
            }
            double n = a + b + c + d;
            return (4 * b * c) / pow(n, 2);
        }


        template <typename T>
        double lanceWilliams(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a = 0.0,
            b = 0.0,
            c = 0.0;
            for (std::size_t i = 0; i < d1; ++i) {
                if (obj1[i] && obj2[i]) {
                    ++a;
                } else if (obj1[i] && !obj2[i]) {
                    ++b;
                } else if (!obj1[i] && obj2[i]) {
                    ++c;
                }
            }
            return (b + c) / ((2 * a) + b + c);
        }


        template <typename T>
        double brayCurtis(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a = 0.0,
            b = 0.0,
            c = 0.0;
            for (std::size_t i = 0; i < d1; ++i) {
                if (obj1[i] && obj2[i]) {
                    ++a;
                } else if (obj1[i] && !obj2[i]) {
                    ++b;
                } else if (!obj1[i] && obj2[i]) {
                    ++c;
                }
            }
            return (b + c) / ((2 * a) + b + c);
        }


        template <typename T>
        double hellinger(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a = 0.0,
            b = 0.0,
            c = 0.0;
            for (std::size_t i = 0; i < d1; ++i) {
                if (obj1[i] && obj2[i]) {
                    ++a;
                } else if (obj1[i] && !obj2[i]) {
                    ++b;
                } else if (!obj1[i] && obj2[i]) {
                    ++c;
                }
            }
            double denominator = sqrt((a + b) * (a + c));
            double value = a / denominator;
            return 2 * sqrt(1 - value);
        }


        template <typename T>
        double chord(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a = 0.0,
            b = 0.0,
            c = 0.0;
            for (std::size_t i = 0; i < d1; ++i) {
                if (obj1[i] && obj2[i]) {
                    ++a;
                } else if (obj1[i] && !obj2[i]) {
                    ++b;
                } else if (!obj1[i] && obj2[i]) {
                    ++c;
                }
            }
            double denominator = sqrt((a + b) * (a + c));
            double value = 1 - (a / denominator);
            return sqrt(2 * value);
        }
    }
}

#endif /* DISTANCE_H */
