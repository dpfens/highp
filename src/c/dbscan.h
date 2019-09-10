#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

const long long int DBSCAN_UNCLASSIFIED = -2;
const long long int DBSCAN_NOISE = -1;

long long int * dbscan(long double ** data, unsigned long long int sample_count, unsigned long long int dimension_count, long double eps, long long int min_points, long double (*distance_func)(long double *, long double *, unsigned long long int), bool verbose) {
    size_t cluster_size = sizeof(long long int);
    long long int * clusters = (long long int *)calloc(sample_count, cluster_size);
    if (clusters == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < sample_count; ++i) {
        clusters[i] = DBSCAN_UNCLASSIFIED;
    }
    unsigned long long int distance_matrix_size = sample_count * sample_count;
    bool * neighbors = malloc(sizeof(bool) * distance_matrix_size);
    long long int * neighbor_counts = calloc(sample_count, cluster_size);
    if (neighbors == NULL) {
        return NULL;
    }

    long double * point;
    long double * other_point;
    long double distance;
    size_t i = 0;

    for (i = 0; i < sample_count; ++i) {
        point = data[i];
        for (size_t j = i; j < sample_count; ++j) {
            bool is_neighbor = true;
            if (i == j) {
                neighbors[i * sample_count + j] = is_neighbor;
                neighbors[j * sample_count + i] = is_neighbor;
                ++neighbor_counts[i];
                continue;
            }
            other_point = data[j];
            distance = distance_func(point, other_point, dimension_count);
            if (distance >= eps) {
                is_neighbor = false;
            } else {
              ++neighbor_counts[i];
              ++neighbor_counts[j];
            }
            neighbors[i * sample_count + j] = is_neighbor;
            neighbors[j * sample_count + i] = is_neighbor;
        }
    }

    long long int current_cluster;
    long long int cluster_id = 0;

    bool * seeds = malloc(sizeof(bool) * sample_count);
    bool has_seeds;

    bool expanded_cluster;
    i = 0;
    for (i = 0; i < sample_count; ++i) {
        current_cluster = clusters[i];
        if (current_cluster != DBSCAN_UNCLASSIFIED) {
            continue;
        }
        if (neighbor_counts[i] < min_points) {
            clusters[i] = DBSCAN_NOISE;
            expanded_cluster = false;
        } else {
            expanded_cluster = true;
            has_seeds = true;

            size_t j = 0;
            for (j = 0; j < sample_count; ++j) {
                if (neighbors[i * sample_count + j]){
                    clusters[j] = cluster_id;
                    seeds[j] = true;
                    if (!has_seeds) {
                        has_seeds = true;
                    }
                } else {
                    seeds[j] = false;
                }
            }

            while (has_seeds) {
                j = 0;

                for (j = 0; j <sample_count; ++j) {
                    if (!seeds[j]) {
                        continue;
                    }
                    if (neighbor_counts[j] < min_points) {
                        seeds[j] = false;
                        continue;
                    }
                    size_t k = 0;
                    for (k = 0; k < sample_count; ++k) {
                        if (!neighbors[j * sample_count + k]) {
                            continue;
                        }
                        if (clusters[k] == DBSCAN_UNCLASSIFIED || clusters[k] == DBSCAN_NOISE) {
                            if (clusters[k] == DBSCAN_UNCLASSIFIED) {
                                seeds[k] = true;
                            }
                            if (verbose) {
                                printf("Assigning row #%lu to %lli", k, cluster_id);
                            }
                            clusters[k] = cluster_id;
                        }
                    }
                    seeds[j] = false;
                }

                has_seeds = false;

                size_t j = 0;
                #pragma omp parallel for private(j)
                for(j = 0; j < sample_count; ++j) {
                    if (!has_seeds && seeds[j]) {
                        has_seeds = true;
                    }
                }
            }
        }
        if (expanded_cluster) {
          ++cluster_id;
        }
    }
    free(seeds);
    free(neighbors);
    free(neighbor_counts);
    return clusters;
}
