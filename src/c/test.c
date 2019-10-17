#include "dbscan.c"
#include "tdbscan.h"
#include "distance.c"

int main() {
  unsigned long long int sample_count = 141;
  long double raw_data[141] = {931.0, 931.0, 932.0, 932.0, 932.0, 932.0, 932.0, 932.0, 933.0, 933.0, 933.0, 933.0, 933.0, 933.0, 933.0, 933.0, 933.0, 934.0, 934.0, 934.0, 934.0, 934.0, 934.0, 934.0, 934.0, 934.0, 934.0, 935.0, 935.0, 935.0, 935.0, 935.0, 936.0, 936.0, 936.0, 936.0, 936.0, 936.0, 937.0, 938.0, 938.0, 938.0, 938.0, 938.0, 939.0, 939.0, 939.0, 939.0, 939.0, 940.0, 940.0, 940.0, 940.0, 941.0, 941.0, 941.0, 942.0, 942.0, 942.0, 943.0, 944.0, 944.0, 945.0, 945.0, 945.0, 945.0, 946.0, 946.0, 947.0, 947.0, 947.0, 948.0, 948.0, 948.0, 949.0, 949.0, 949.0, 949.0, 949.0, 950.0, 950.0, 950.0, 950.0, 951.0, 951.0, 952.0, 953.0, 953.0, 955.0, 955.0, 965.0, 966.0, 966.0, 966.0, 966.0, 967.0, 968.0, 968.0, 968.0, 968.0, 969.0, 969.0, 970.0, 970.0, 970.0, 971.0, 971.0, 972.0, 972.0, 972.0, 973.0, 973.0, 974.0, 980.0, 980.0, 981.0, 981.0, 981.0, 982.0, 983.0, 983.0, 983.0, 983.0, 984.0, 984.0, 994.0, 994.0, 996.0, 1002.0, 1007.0, 1007.0, 1007.0, 1007.0, 1008.0, 1009.0, 1009.0, 1010.0, 1028.0, 1030.0, 1061.0, 1078.0};
  unsigned long long int dimension_count = 1;
  long double eps = 2;
  long long int min_points = 3;
  bool verbose = true;

  long double ** data = malloc(sizeof(long double) * sample_count);
  for (int i = 0; i < sample_count; ++i) {
    data[i] = &raw_data[i];
  }
  long long int * dbscan_clusters = dbscan(data, sample_count, dimension_count, eps, min_points, euclidean_distance, verbose);
  for(int i = 0; i < sample_count; ++i) {
    printf("Row #%u: %Lf Cluster #%lli\n", i, raw_data[i], dbscan_clusters[i]);
  }
  free(dbscan_clusters);

  long double c_eps = 5.0;
  struct TDBSCAN_Results * results = tdbscan(data, sample_count, dimension_count, eps, c_eps, min_points, euclidean_distance, verbose);
  for (int i = 0; i < sample_count; ++i) {
    printf("Row #%u: %Lf TDBSCAN Cluster #%lli of type: %lli\n", i, raw_data[i], results->clusters[i], results->cluster_types[i]);
  }
  free_tdbscan_results(results);
  free(data);
}
