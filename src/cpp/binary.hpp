#include <math.h>
#include <algorithm>

namespace similarity {

    namespace binary {

        template <typename T>
        double jaccard(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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

            return a / (a + b + c);
        }

        template <typename T>
        double dice(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double aa = 2.0 * a;
            return aa / (aa + b + c);
        }

        template <typename T>
        double jaccard3w(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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

            double aaa = 3.0 * a;
            return aaa / (aaa + b + c);
        }

        template <typename T>
        double neiLi(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            return (2.0 * a) / (a + b + a + c);
        }

        template <typename T>
        double sokalSneath(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            return a / (a + (2.0 * b) + (2 * c));
        }

        template <typename T>
        double sokalMichener(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            return a + d / n;
        }

        template <typename T>
        double sokalSneath2(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            return (2 * (a + d)) / n;
        }


        template <typename T>
        double rogerTanimoto(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double denominator = a + (2 * (b + c)) + d;
            return (a + d) / denominator;
        }


        template <typename T>
        double faith(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = a + (0.5 * d);
            return numerator / n;
        }


        template <typename T>
        double gowerLegendre(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = a + d;
            double denominator = a + (0.5 * (b + c)) + d;
            return numerator / denominator;
        }


        template <typename T>
        double intersection(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0;
            for (std::size_t i = 0; i < d1; ++i) {
                if (obj1[i] && obj2[i]) {
                    ++a;
                }
            }
            return a;
        }


        template <typename T>
        double innerProduct(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
            d = 0.0;
            for (std::size_t i = 0; i < d1; ++i) {
                if (obj1[i] && obj2[i]) {
                    ++a;
                } else if (!obj1[i] && !obj2[i]) {
                    ++d;
                }
            }
            return a + d;
        }


        template <typename T>
        double russellRao(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            return a / n;
        }


        template <typename T>
        double cosine(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            return a / denominator;
        }


        template <typename T>
        double gilbertWells(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            return log(a) - log(n) - log((a + b) / n) - log((a + c) / n);
        }


        template <typename T>
        double ochai(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            return a / denominator;
        }


        template <typename T>
        double forbesi(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double denominator = (a + b) * (a + c);
            return (n * a) / denominator;
        }


        template <typename T>
        double fossum(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = n * pow(a - 0.5, 2);
            double denominator = (a + b) * (a + c);
            return numerator / denominator;
        }


        template <typename T>
        double sorgenfrei(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = pow(a, 2);
            double denominator = (a + b) * (a + c);
            return numerator / denominator;
        }


        template <typename T>
        double mountford(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = a;
            double denominator = 0.5 * ((a * b) + (a * c)) + (b * c);
            return numerator / denominator;
        }


        template <typename T>
        double otsuka(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = a;
            double denominator = pow((a + b)* (a + c), 0.5);
            return numerator / denominator;
        }


        template <typename T>
        double mcconnaughey(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = pow(a, 2) - (b * c);
            double denominator = (a + b) * (a + c);
            return numerator / denominator;
        }


        template <typename T>
        double tarwid(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double na = n * a;
            double coefficient = (a + b) * (a + c);
            double numerator = na - coefficient;
            double denominator = na + coefficient;
            return numerator / denominator;
        }


        template <typename T>
        double kulczynski2(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = (a / 2) * (2 * a  + b + c);
            double denominator = (a + b) * (a + c);
            return numerator / denominator;
        }


        template <typename T>
        double driverKroeber(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            return (a / 2) * ((1 / a + b) + (1 / a + c));
        }


        template <typename T>
        double johnson(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            return (a / (a + b)) + (a / (a + c));
        }


        template <typename T>
        double dennis(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = (a * d) - (b * c);
            double denominator = sqrt(n * ((a + b) * (a + c)) );
            return numerator / denominator;
        }


        template <typename T>
        double simpson(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double ab = a + b;
            double ac = a + c;
            double denominator = std::min(ab, ac);
            return a / denominator;
        }


        template <typename T>
        double braunBanquet(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double ab = a + b;
            double ac = a + c;
            double denominator = std::max(ab, ac);
            return a / denominator;
        }


        template <typename T>
        double fagerMcgowan(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double ab = a + b;
            double ac = a + c;
            double value1 = a / sqrt(ab * ac);
            double value2 = std::max(ab, ac) / 2;
            return value1 - value2;
        }


        template <typename T>
        double forbes2(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double ab = a + b;
            double ac = a + c;
            double numerator = (n * a) - ab * ac;
            double denominator = n * std::min(ab, ac) - ab * ac;
            return numerator / denominator;
        }


        template <typename T>
        double sokalSneath4(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double ab = a + b;
            double ac = a + c;
            double numerator = (a  / (a + b)) + (a / (a + c)) + (d / (b + d)) + (d / (c + d));
            return numerator / 4;
        }


        template <typename T>
        double gower(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = a + d;
            double denominator = sqrt((a + b) * (a + c) * (b + d) * (c + d));
            return numerator / denominator;
        }


        template <typename T>
        double pearson1(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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

            double numerator = n * pow((a * d) - (b * c), 2);
            double denominator = (a + b) * (a + c) * (c + d) * (b + d);
            double chi = numerator / denominator;
            return pow(chi, 2);
        }


        template <typename T>
        double pearson2(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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

            double numerator = n * pow((a * d) - (b * c), 2);
            double denominator = (a + b) * (a + c) * (c + d) * (b + d);
            double chi = numerator / denominator;
            return pow(chi / (n + chi), 0.5);
        }


        template <typename T>
        double pearson3(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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

            double numerator = (a * d) - (b * c);
            double denominator = sqrt((a + b) * (a + c) * (b + d) * (c + d));
            double p = numerator / denominator;
            return pow(p / (n + p), 0.5);
        }


        template <typename T>
        double pearsonHeron1(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = (a * d) - (b * c);
            double denominator = sqrt((a + b) * (a + c) * (b + d) * (c + d));
            return numerator / denominator;
        }


        template <typename T>
        double pearsonHeron2(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double bc = b * c;
            double numerator = M_PI * sqrt(bc);
            double denominator = sqrt(a * d) + sqrt(bc);
            return cos(numerator / denominator);
        }


        template <typename T>
        double sokalSneath3(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            return (a + d) / (b + c);
        }


        template <typename T>
        double sokalSneath5(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = a * d;
            double denominator = (a + b) * (a + c) * (b + d) * pow(c + d, 0.5);
            return numerator / denominator;
        }


        template <typename T>
        double cole(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = sqrt(2) * (ad - bc);
            double denominator = sqrt(pow(ad - bc, 2) - (a + b) * (a + c) * (b + d) * (c + d));
            return numerator / denominator;
        }


        template <typename T>
        double ochai2(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = a * d;
            double denominator = sqrt((a + b) * (a + c) * (b + d) * (c + d));
            return numerator / denominator;
        }


        template <typename T>
        double yuleq(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = ad - bc;
            double denominator = ad + bc;
            return numerator / denominator;
        }


        template <typename T>
        double yulew(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double ad = sqrt(a * d);
            double bc = sqrt(b * c);
            double numerator = ad - bc;
            double denominator = ad + bc;
            return numerator / denominator;
        }


        template <typename T>
        double kulczynski1(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            return a / (b + c);
        }


        template <typename T>
        double tanimoto(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            return a / ((a + b + a + c) - a);
        }


        template <typename T>
        double disperson(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = (a * d) - (b * c);
            return numerator / pow(n, 2);
        }


        template <typename T>
        double hamann(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = (a + d) - (b + c);
            return numerator / n;
        }


        template <typename T>
        double michael(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = 4 * ((a * d) - (b * c));
            double denominator = pow(a + d, 2) + pow(b + c, 2);
            return numerator / denominator;
        }


        template <typename T>
        double goodmanKruskal(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double sigma = std::max(a, b) + std::max(c, d) + std::max(a, c) + std::max(b, d);
            double sigma_m = std::max(a + c, b + d) + std::max(a + b, c + d);
            return (sigma - sigma_m) / (2*n - sigma_m);
        }


        template <typename T>
        double anderberg(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double sigma = std::max(a, b) + std::max(c, d) + std::max(a, c) + std::max(b, d);
            double sigma_m = std::max(a + c, b + d) + std::max(a + b, c + d);
            return (sigma - sigma_m) / (2*n);
        }


        template <typename T>
        double baroniUrbaniBuser1(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double ad = sqrt(a * d);
            double numerator = ad + a;
            double denominator = ad + a + b + c;
            return numerator / denominator;
        }


        template <typename T>
        double baroniUrbaniBuser2(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double ad = sqrt(a * d);
            double bc = b + c;
            double numerator = ad + a - bc;
            double denominator = ad + a + bc;
            return numerator / denominator;
        }


        template <typename T>
        double pierce(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double ab = a * b;
            double bc = b * c;
            double numerator = ab + bc;
            double denominator = ab + (2 * bc) + c * d;
            return numerator / denominator;
        }


        template <typename T>
        double eyraud(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double ab = a + b;
            double ac = a + c;
            double numerator = pow(n, 2) * (n * a - (ab * ac));
            double denominator = ab * ac * (b + d) * (c + d);
            return numerator / denominator;
        }


        template <typename T>
        double tarantula(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = a * (c + d);
            double denominator = c * (a + b);
            return numerator / denominator;
        }


        template <typename T>
        double ample(std::vector<T> obj1, std::vector<T> obj2) {

            std::size_t d1 = obj1.size(), d2 = obj2.size();

            if (d1 != d2) {
                return 0.0;
            }
            double a  = 0.0,
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
            double numerator = a * (c + d);
            double denominator = c * (a + b);
            return abs(numerator / denominator);
        }

    }
}
