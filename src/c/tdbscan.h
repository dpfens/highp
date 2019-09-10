#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

const long long int TDBSCAN_NOISE = -1;
const long long int TDBSCAN_STOP = 0;
const long long int TDBSCAN_MOVE = 1;

struct TDBSCAN_Results {
  long long int * clusters;
  long long int * cluster_types;
};

void free_tdbscan_results(struct TDBSCAN_Results * results) {
    free(results->clusters);
    free(results->cluster_types);
    free(results);
}

struct TDBSCAN_Results * tdbscan(long double ** data, unsigned long long int sample_count, unsigned long long int dimension_count, long double eps, long double c_eps, long long int min_points, long double (*distance_func)(long double *, long double *, unsigned long long int), bool verbose) {
    size_t cluster_size = sizeof(long long int);
    long long int * clusters = (long long int *)calloc(sample_count, cluster_size);
    if (clusters == NULL) {
        return NULL;
    }
    size_t i = 0;
    #pragma omp parallel for private(i)
    for (i = 0; i < sample_count; ++i) {
        clusters[i] = TDBSCAN_NOISE;
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

    for (i = 0; i < sample_count; ++i) {
        point = data[i];
        for (size_t j = i; j < sample_count; ++j) {
            bool is_neighbor = true;
            other_point = data[j];
            distance = distance_func(point, other_point, dimension_count);
            if (distance < eps) {
                is_neighbor = true;
            } else {
                is_neighbor = false;
            }
            neighbors[i * sample_count + j] = is_neighbor;
            neighbors[j * sample_count + i] = is_neighbor;
            ++neighbor_counts[i];
            if (distance > c_eps) {
                break;
            }
        }
    }

    long long int cluster_id = 0;
    bool * seeds = malloc(sizeof(bool) * sample_count);

    size_t j = 0;
    unsigned long long int max_id = 0;
    while (max_id < sample_count) {
      printf("MAX ID: %lli: neighbors: %lli\n", max_id, neighbor_counts[max_id]);
        if (neighbor_counts[max_id] < min_points) {
            ++cluster_id;
        }
        // expand cluster
        if (verbose) {
            printf("Assigning %llu to cluster #%lli\n", max_id, cluster_id);
        }
        clusters[max_id] = cluster_id;
        for (i = max_id; i < sample_count; ++i) {
            if (!neighbors[max_id * sample_count + i]) {
                seeds[i] = false;
                break;
            }
            seeds[i] = true;
        }
        for (i = max_id; i < sample_count; ++i) {
            if (!seeds[i]) {
                continue;
            }
            if (i > max_id) {
                max_id = i;
            }
            if (neighbor_counts[max_id] >= min_points) {
                if (verbose) {
                    printf("Assigning %llu to cluster #%lli\n", max_id, cluster_id);
                }
                clusters[max_id] = cluster_id;
                for (j = max_id; j < sample_count; ++j) {
                    if (neighbors[max_id * sample_count + j]) {
                        seeds[j] = true;
                    } else {
                        seeds[j] = false;
                        break;
                    }
                }
            }
            if (clusters[max_id] == TDBSCAN_NOISE) {
                if (verbose) {
                    printf("Assigning %llu to cluster #%lli\n", max_id, cluster_id);
                }
                clusters[max_id] = cluster_id;
            }
        }
        ++max_id;
    }
    free(seeds);
    free(neighbors);
    free(neighbor_counts);

    // merge clusters with overlapping min & max times
    long long int * min_indices = (long long int *)calloc(cluster_id, cluster_size);
    long long int * max_indices = (long long int *)calloc(cluster_id, cluster_size);

    for (i = 0; i < cluster_id; ++i) {
      min_indices[i] = LLONG_MAX;
      max_indices[i] = LLONG_MIN;
    }

    for (i = 0; i < sample_count; ++i) {
      if (i < min_indices[clusters[i]]) {
        min_indices[clusters[i]] = i;
      } else if (i > max_indices[clusters[i]]) {
        max_indices[clusters[i]] = i;
      }
    }

    for (i = 0; i < cluster_id - 1; ++i) {
      if(max_indices[i] < min_indices[i + 1]) {
        continue;
      }

      for (j = 0; j < sample_count; ++j) {
        if (clusters[j] == i) {
          clusters[j] = i + 1;
        }
      }
    }

    free(min_indices);
    free(max_indices);

    // build results
    long long int cluster;
    long long int previous_cluster = clusters[0];
    bool is_last_point;
    long long int cluster_start = 0;
    long long int cluster_stop = 0;
    long long int cluster_type;

    struct TDBSCAN_Results * results = malloc(sizeof(struct TDBSCAN_Results));
    long long int * cluster_types = malloc(sizeof(long long int) * sample_count);
    for(i = 1; i < sample_count; ++i) {
        cluster = clusters[i];
        is_last_point = i == sample_count - 1;
        if (previous_cluster != cluster || is_last_point) {
            if (is_last_point) {
                cluster_stop = i + 1;
            } else {
                cluster_stop = i;
            }
            if (cluster_stop - cluster_start <= min_points) {
                cluster_type = TDBSCAN_MOVE;
            } else {
                cluster_type = TDBSCAN_STOP;
            }
            for (j = cluster_start; j < cluster_stop; ++j) {
                cluster_types[j] = cluster_type;
            }
            cluster_start = cluster_stop;
        }
        previous_cluster = cluster;
    }
    results->clusters = clusters;
    results->cluster_types = cluster_types;
    return results;
}
