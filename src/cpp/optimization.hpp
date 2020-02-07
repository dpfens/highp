#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <tuple>

namespace optimization {

    template <typename T>
    std::tuple<T, size_t, size_t> max_subarray(std::vector<T> & data) {
        T best_sum = 0, current_sum = 0, value;
        size_t best_start = 0, best_end = 0, current_start;
        size_t i = 0;
        size_t n = data.size();
        for (i = 0; i < n; ++i) {
            value = data[i];
            if (current_sum <= 0) {
                current_start = i;
                current_sum = value;
            } else {
                current_sum += value;
            }

            if (current_sum > best_sum) {
                best_sum = current_sum;
                best_start = current_start;
                best_end = i + 1;
            }
        }
        return std::make_tuple(best_sum, best_start, best_end);
    }
}

#endif /* OPTIMIZATION_H */
