#ifndef RANKING_H
#define RANKING_H

#include <math.h>

namespace ranking {

    template <typename T>
    std::vector<T> rankify(std::vector<T> & X) {
        std::size_t N = X.size();
        std::vector<T> rankings(N);
        std::size_t i = 0;
        #pragma omp parallel for private(i) shared(rankings)
        for(std::size_t i = 0; i < N; i++) {
            size_t r = 1, s = 1;

            for(std::size_t j = 0; j < i; j++) {
                if (X[j] < X[i] ) r++;
                if (X[j] == X[i] ) s++;
            }

            for (std::size_t j = i + 1; j < N; j++) {
                if (X[j] < X[i] ) r++;
                if (X[j] == X[i] ) s++;
            }
            rankings[i] = r + (s - 1) * 0.5;
        }
        return rankings;
    }

    template <class T, class T2>
    T2 spearman(std::vector<T> &X, std::vector<T> &Y) {
        std::size_t n = X.size();
        T2 sum_X = 0, sum_Y = 0, sum_XY = 0, squareSum_X = 0, squareSum_Y = 0;

        for (std::size_t i = 0; i < n; i++) {
            sum_X = sum_X + X[i];
            sum_Y = sum_Y + Y[i];
            sum_XY += X[i] * Y[i];
            squareSum_X += X[i] * X[i];
            squareSum_Y += Y[i] * Y[i];
        }

        T2 corr = (T2)(n * sum_XY - sum_X * sum_Y)
        / sqrt((n * squareSum_X - sum_X * sum_X)
        * (n * squareSum_Y - sum_Y * sum_Y));

        return corr;
    }
}


#endif /* RANKING_H */
