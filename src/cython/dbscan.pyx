from libc.stdlib cimport malloc, free

cdef extern from "time.h":
    long int time(int)


cdef extern from "../c/dbscan.h":
    long long int * dbscan(long double ** data, unsigned long long int sample_count, unsigned long long int dimension_count, long double eps, long long int min_points, long double (*distance_func)(long double *, long double *, unsigned long long int), bint verbose)

cdef extern from "../c/tdbscan.h":
    struct TDBSCAN_Results:
        long long int * clusters
        long long int * cluster_types

    void free_tdbscan_results(TDBSCAN_Results * results)

    TDBSCAN_Results * tdbscan(long double ** data, unsigned long long int sample_count, unsigned long long int dimension_count, long double eps, long double c_eps, long long int min_points, long double (*distance_func)(long double *, long double *, unsigned long long int), bint verbose)


cdef extern from "../c/distance.h":
    long double euclidean_distance(long double * point1, long double * point2, unsigned long long int dimension_count)
    long double great_circle_distance(long double * point1, long double * point2, unsigned long long int dimension_count)


cdef class DBSCAN:
    cdef double eps
    cdef unsigned int min_points
    cdef str metric
    cdef long double (*distance_func)(long double *, long double *, unsigned long long int)


    def __cinit__(DBSCAN self, double eps, int min_points=2, str metric='euclidean'):
        self.eps = eps
        self.min_points = min_points
        self.metric = metric
        if metric == 'euclidean':
            self.distance_func = euclidean_distance
        else:
            self.distance_func = great_circle_distance

    cpdef list fit_predict(DBSCAN self, list X, list y=None):
        cdef bint verbose = False
        cdef unsigned long long int sample_count = len(X)
        cdef unsigned long long int dimension_count =  len(X[0])
        cdef long double * raw_data = <long double *> malloc(sample_count * dimension_count * sizeof(long double))
        cdef long double ** data = <long double **> malloc(sample_count * sizeof(long double))


        cdef unsigned long long int i = 0
        cdef list sample
        cdef unsigned long long int j

        for i in range(sample_count):
            sample = X[i]
            j = 0
            data[i] = &raw_data[i * dimension_count]
            for j in range(dimension_count):
                raw_data[i * dimension_count + j] = <long double>sample[j]

        cdef long long int * raw_clusters = dbscan(data, sample_count, dimension_count, self.eps, self.min_points, self.distance_func, verbose)
        free(data)
        free(raw_data)
        cdef list clusters = list()
        i = 0

        cdef long long int cluster_assignment
        for i in range(sample_count):
            cluster_assignment = raw_clusters[i]
            clusters.append(cluster_assignment)
        free(raw_clusters)
        return clusters


cdef class TDBSCAN:
    cdef double eps
    cdef unsigned int min_points
    cdef double c_eps
    cdef str metric
    cdef long double (*distance_func)(long double *, long double *, unsigned long long int)


    def __cinit__(DBSCAN self, double eps, double c_eps, int min_points=2, str metric='euclidean'):
        self.eps = eps
        self.min_points = min_points
        self.c_eps = c_eps
        self.metric = metric
        if metric == 'euclidean':
            self.distance_func = euclidean_distance
        else:
            self.distance_func = great_circle_distance

    cpdef list fit_predict(TDBSCAN self, list X, list y=None):
        cdef bint verbose = False
        cdef unsigned long long int sample_count = len(X)
        cdef unsigned long long int dimension_count =  len(X[0])
        cdef long double * raw_data = <long double *> malloc(sample_count * dimension_count * sizeof(long double))
        cdef long double ** data = <long double **> malloc(sample_count * sizeof(long double))

        cdef unsigned long long int i = 0
        cdef list sample
        cdef unsigned long long int j

        for i in range(sample_count):
            sample = X[i]
            j = 0
            data[i] = &raw_data[i * dimension_count]
            for j in range(dimension_count):
                raw_data[i * dimension_count + j] = <long double>sample[j]

        cdef TDBSCAN_Results * results = tdbscan(data, sample_count, dimension_count, self.eps, self.c_eps, self.min_points, self.distance_func, verbose)
        free(data)
        free(raw_data)

        cdef list clusters = list()
        cdef list cluster_types = list()

        i = 0
        cdef long long int cluster_assignment
        for i in range(sample_count):
            cluster_assignment = results.clusters[i]
            clusters.append(cluster_assignment)
            cluster_types.append(results.cluster_types[i]);
        free_tdbscan_results(results)
        return clusters
