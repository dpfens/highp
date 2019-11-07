#include <set>
#include <vector>
#include <map>
#include <math.h>

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

    namespace fuzzy {

        template <typename T>
        double dengfeng_chuntian(std::vector<std::map<T, double> > &data, T a, T b) {
            double output = 0.0;
            double p = 2.0;
            size_t n = data.size();
            for(size_t i = 0; i < n; ++i) {
                double a_membership;
                double a_nonmembership;
                if (data[i].find(a) == data[i].end()) {
                    a_membership = 0.0;
                    a_nonmembership = 1.0;
                } else {
                    a_membership = data[i][a];
                    a_nonmembership = 1.0 - a_membership;
                }
                double m_a = (a_membership + 1.0 - a_nonmembership) / 2.0;
                double b_membership;
                double b_nonmembership;
                if (data[i].find(b) == data[i].end()) {
                    b_membership = 0.0;
                    b_nonmembership = 1.0;
                } else {
                    b_membership = data[i][b];
                    b_nonmembership = 1.0 - b_membership;
                }
                double m_b = (b_membership + 1.0 - b_nonmembership) / 2.0;
                output += pow(abs(m_a - m_b), p);
            }
            output = pow(output, 1/p);
            double multiplier = 1.0 - (1.0 / pow(n, 1/p));
            return multiplier * output;
        }

        template <typename T>
        double liang_shi(std::vector<std::map<T, double> > &data, T a, T b) {
            double output = 0.0;
            double p = 2.0;
            size_t n = data.size();
            for(size_t i = 0; i < n; ++i) {
                double a_membership;
                double a_nonmembership;
                if (data[i].find(a) == data[i].end()) {
                    a_membership = 0.0;
                    a_nonmembership = 1.0;
                } else {
                    a_membership = data[i][a];
                    a_nonmembership = 1.0 - a_membership;
                }
                double b_membership;
                double b_nonmembership;
                if (data[i].find(b) == data[i].end()) {
                    b_membership = 0.0;
                    b_nonmembership = 1.0;
                } else {
                    b_membership = data[i][b];
                    b_nonmembership = 1.0 - b_membership;
                }
                double phi_tab = (a_membership - b_membership) / 2.0;
                double phi_fab = ((1.0 - b_nonmembership) - (1.0 - a_nonmembership)) / 2.0;
                output += pow(abs(phi_tab + phi_fab), p);
            }
            output = pow(output, 1.0 / p);
            double multiplier = 1.0 - (1.0 / pow(n, 1 / p));
            return multiplier * output;
        }

        template <typename T>
        double hwang_yang_hung_g(T m, T n) {
            if (m == 0 && n == 0) {
                return 1.0;
            }
            double mn = m * n;
            double m_sq = m * m;
            double n_sq = n * n;
            return mn / (m_sq + n_sq - mn);
        }

        template <typename T>
        double hwang_yang_hung(std::vector<std::map<T, double> > &data, T a, T b) {
            double output = 0.0;
            size_t n = data.size();
            double coefficient = 1.0 / (3.0 * n);
            for(size_t i = 0; i < n; ++i) {

                double a_membership;
                double a_nonmembership;
                if (data[i].find(a) == data[i].end()) {
                    a_membership = 0.0;
                    a_nonmembership = 1.0;
                } else {
                    a_membership = data[i][a];
                    a_nonmembership = 1.0 - a_membership;
                }
                double b_membership;
                double b_nonmembership;
                if (data[i].find(b) == data[i].end()) {
                    b_membership = 0.0;
                    b_nonmembership = 1.0;
                } else {
                    b_membership = data[i][b];
                    b_nonmembership = 1.0 - b_membership;
                }
                double g_value = hwang_yang_hung_g(a_membership, b_membership);
                double g_normalized_membership = hwang_yang_hung_g(0.5 * (1.0 + a_membership - a_nonmembership), 0.5 * (1.0 + b_membership - b_nonmembership));
                double value = g_value + g_value + g_normalized_membership;
                output += value;
            }
            return coefficient * output;
        }
    }
}
