#include "fuzzy.cpp"
#include "distance.cpp"
#include "moving.cpp"
#include "similarity.cpp"
#include "binary.hpp"
#include "kmeans.cpp"
#include "fuzzy_pack.cpp"

template <class T, class T2>
void print_map(std::map<T, T2> &data) {
    std::cout << "{";
    for (const auto &element : data) {
        std::cout << element.first << ": " << element.second << ", ";
    }
    std::cout << "}";
}

template <class T>
void print_vector(std::vector<T> &data) {
    std::cout << "[";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i] << ", ";
    }
    std::cout << "]\n";
}

int main() {
    std::vector<std::vector<double> > data = {{931.0}, {931.0}, {932.0}, {932.0}, {932.0}, {932.0}, {932.0}, {932.0}, {933.0}, {933.0}, {933.0}, {933.0}, {933.0}, {933.0}, {933.0}, {933.0}, {933.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {934.0}, {935.0}, {935.0}, {935.0}, {935.0}, {935.0}, {936.0}, {936.0}, {936.0}, {936.0}, {936.0}, {936.0}, {937.0}, {938.0}, {938.0}, {938.0}, {938.0}, {938.0}, {939.0}, {939.0}, {939.0}, {939.0}, {939.0}, {940.0}, {940.0}, {940.0}, {940.0}, {941.0}, {941.0}, {941.0}, {942.0}, {942.0}, {942.0}, {943.0}, {944.0}, {944.0}, {945.0}, {945.0}, {945.0}, {945.0}, {946.0}, {946.0}, {947.0}, {947.0}, {947.0}, {948.0}, {948.0}, {948.0}, {949.0}, {949.0}, {949.0}, {949.0}, {949.0}, {950.0}, {950.0}, {950.0}, {950.0}, {951.0}, {951.0}, {952.0}, {953.0}, {953.0}, {955.0}, {955.0}, {965.0}, {966.0}, {966.0}, {966.0}, {966.0}, {967.0}, {968.0}, {968.0}, {968.0}, {968.0}, {969.0}, {969.0}, {970.0}, {970.0}, {970.0}, {971.0}, {971.0}, {972.0}, {972.0}, {972.0}, {973.0}, {973.0}, {974.0}, {980.0}, {980.0}, {981.0}, {981.0}, {981.0}, {982.0}, {983.0}, {983.0}, {983.0}, {983.0}, {984.0}, {984.0}, {994.0}, {994.0}, {996.0}, {1002.0}, {1007.0}, {1007.0}, {1007.0}, {1007.0}, {1008.0}, {1009.0}, {1009.0}, {1010.0}, {1028.0}, {1030.0}, {1061.0}, {1078.0}};
    std::vector<std::vector<double> > other_data = {{7344.0}, {7380.0}, {7392.0}, {7451.0}, {7466.0}, {7478.0}, {7493.0}, {7499.0}, {7499.0}, {7510.0}, {7543.0}, {7563.0}, {7569.0}, {7569.0}, {7580.0}, {7591.0}, {7609.0}, {7620.0}, {7623.0}, {7631.0}, {7638.0}, {7645.0}, {7663.0}, {7665.0}, {7667.0}, {7686.0}, {7691.0}, {7701.0}, {7701.0}, {7702.0}, {7735.0}, {7750.0}, {7755.0}, {7760.0}, {7777.0}, {7790.0}, {7796.0}, {7797.0}, {7805.0}, {7809.0}, {7811.0}, {7814.0}, {7819.0}, {7820.0}, {7821.0}, {7828.0}, {7833.0}, {7849.0}, {7853.0}, {7853.0}, {7862.0}, {7874.0}, {7877.0}, {7878.0}, {7880.0}, {7886.0}, {7891.0}, {7894.0}, {7896.0}, {7897.0}, {7899.0}, {7900.0}, {7904.0}, {7929.0}, {7945.0}, {7953.0}, {7958.0}, {7961.0}, {7963.0}, {7964.0}, {7970.0}, {7978.0}, {7998.0}, {7998.0}, {7999.0}, {8021.0}, {8021.0}, {8025.0}, {8033.0}, {8056.0}, {8062.0}, {8063.0}, {8070.0}, {8074.0}, {8110.0}, {8113.0}, {8118.0}, {8119.0}, {8125.0}, {8137.0}, {8151.0}, {8151.0}, {8152.0}, {8169.0}, {8192.0}, {8214.0}, {8237.0}, {8249.0}, {8268.0}, {8275.0}, {8278.0}, {8284.0}, {8285.0}, {8303.0}, {8304.0}, {8308.0}, {8322.0}, {8345.0}, {8352.0}, {8361.0}, {8365.0}, {8370.0}, {8380.0}, {8383.0}, {8394.0}, {8416.0}, {8445.0}, {8454.0}, {8457.0}, {8490.0}, {8506.0}, {8512.0}, {8520.0}, {8533.0}, {8540.0}, {8545.0}, {8563.0}, {8569.0}, {8590.0}, {8611.0}, {8810.0}, {8834.0}, {8850.0}, {8858.0}, {8882.0}, {8895.0}, {8896.0}, {8904.0}, {9148.0}, {9347.0}, {9419.0}};
    std::vector<double> single_data = {7344.0, 7380.0, 7392.0, 7451.0, 7466.0, 7478.0, 7493.0, 7499.0, 7499.0, 7510.0, 7543.0, 7563.0, 7569.0, 7569.0, 7580.0, 7591.0, 7609.0, 7620.0, 7623.0, 7631.0, 7638.0, 7645.0, 7663.0, 7665.0, 7667.0, 7686.0, 7691.0, 7701.0, 7701.0, 7702.0, 7735.0, 7750.0, 7755.0, 7760.0, 7777.0, 7790.0, 7796.0, 7797.0, 7805.0, 7809.0, 7811.0, 7814.0, 7819.0, 7820.0, 7821.0, 7828.0, 7833.0, 7849.0, 7853.0, 7853.0, 7862.0, 7874.0, 7877.0, 7878.0, 7880.0, 7886.0, 7891.0, 7894.0, 7896.0, 7897.0, 7899.0, 7900.0, 7904.0, 7929.0, 7945.0, 7953.0, 7958.0, 7961.0, 7963.0, 7964.0, 7970.0, 7978.0, 7998.0, 7998.0, 7999.0, 8021.0, 8021.0, 8025.0, 8033.0, 8056.0, 8062.0, 8063.0, 8070.0, 8074.0, 8110.0, 8113.0, 8118.0, 8119.0, 8125.0, 8137.0, 8151.0, 8151.0, 8152.0, 8169.0, 8192.0, 8214.0, 8237.0, 8249.0, 8268.0, 8275.0, 8278.0, 8284.0, 8285.0, 8303.0, 8304.0, 8308.0, 8322.0, 8345.0, 8352.0, 8361.0, 8365.0, 8370.0, 8380.0, 8383.0, 8394.0, 8416.0, 8445.0, 8454.0, 8457.0, 8490.0, 8506.0, 8512.0, 8520.0, 8533.0, 8540.0, 8545.0, 8563.0, 8569.0, 8590.0, 8611.0, 8810.0, 8834.0, 8850.0, 8858.0, 8882.0, 8895.0, 8896.0, 8904.0, 9148.0, 9347.0, 9419.0};
    std::vector<std::vector<std::vector<double> > > sequential_data = {
    {{931.0}, {1869.0}, {2813.0}, {3748.0}, {4693.0}, {5595.0}, {6460.0}, {7344.0}},
    {{936.0}, {1871.0}, {2820.0}, {3750.0}, {4694.0}, {5595.0}, {6461.0}, {7380.0}},
    {{932.0}, {1870.0}, {2814.0}, {3749.0}, {4693.0}, {5595.0}, {6461.0}, {7392.0}},
    {{936.0}, {1868.0}, {2814.0}, {3747.0}, {4693.0}, {5610.0}, {6533.0}, {7451.0}},
    {{933.0}, {1872.0}, {2821.0}, {3750.0}, {4695.0}, {5605.0}, {6534.0}, {7466.0}},
    {{933.0}, {1871.0}, {2816.0}, {3749.0}, {4694.0}, {5596.0}, {6508.0}, {7478.0}},
    {{942.0}, {1878.0}, {2819.0}, {3748.0}, {4694.0}, {5610.0}, {6543.0}, {7493.0}},
    {{938.0}, {1873.0}, {2820.0}, {3750.0}, {4695.0}, {5612.0}, {6552.0}, {7499.0}},
    {{933.0}, {1870.0}, {2816.0}, {3749.0}, {4694.0}, {5596.0}, {6538.0}, {7499.0}},
    {{931.0}, {1869.0}, {2814.0}, {3747.0}, {4692.0}, {5595.0}, {6508.0}, {7510.0}},
    {{943.0}, {1882.0}, {2821.0}, {3762.0}, {4704.0}, {5635.0}, {6587.0}, {7543.0}},
    {{933.0}, {1870.0}, {2815.0}, {3748.0}, {4694.0}, {5595.0}, {6534.0}, {7563.0}},
    {{940.0}, {1873.0}, {2822.0}, {3756.0}, {4696.0}, {5632.0}, {6592.0}, {7569.0}},
    {{935.0}, {1872.0}, {2817.0}, {3748.0}, {4695.0}, {5632.0}, {6590.0}, {7569.0}},
    {{933.0}, {1873.0}, {2820.0}, {3751.0}, {4694.0}, {5616.0}, {6587.0}, {7580.0}},
    {{939.0}, {1876.0}, {2823.0}, {3751.0}, {4697.0}, {5636.0}, {6591.0}, {7591.0}},
    {{934.0}, {1871.0}, {2818.0}, {3749.0}, {4695.0}, {5636.0}, {6593.0}, {7609.0}},
    {{936.0}, {1870.0}, {2815.0}, {3747.0}, {4694.0}, {5599.0}, {6570.0}, {7620.0}},
    {{940.0}, {1874.0}, {2821.0}, {3750.0}, {4695.0}, {5629.0}, {6604.0}, {7623.0}},
    {{942.0}, {1877.0}, {2824.0}, {3763.0}, {4706.0}, {5661.0}, {6636.0}, {7631.0}},
    {{950.0}, {1899.0}, {2864.0}, {3821.0}, {4786.0}, {5716.0}, {6670.0}, {7638.0}},
    {{939.0}, {1873.0}, {2821.0}, {3751.0}, {4697.0}, {5616.0}, {6589.0}, {7645.0}},
    {{940.0}, {1877.0}, {2823.0}, {3762.0}, {4705.0}, {5663.0}, {6656.0}, {7663.0}},
    {{945.0}, {1879.0}, {2823.0}, {3763.0}, {4705.0}, {5654.0}, {6642.0}, {7665.0}},
    {{935.0}, {1871.0}, {2815.0}, {3749.0}, {4695.0}, {5601.0}, {6600.0}, {7667.0}},
    {{932.0}, {1871.0}, {2818.0}, {3749.0}, {4697.0}, {5651.0}, {6654.0}, {7686.0}},
    {{933.0}, {1871.0}, {2817.0}, {3751.0}, {4696.0}, {5647.0}, {6660.0}, {7691.0}},
    {{944.0}, {1881.0}, {2823.0}, {3768.0}, {4740.0}, {5699.0}, {6691.0}, {7701.0}},
    {{948.0}, {1902.0}, {2864.0}, {3822.0}, {4788.0}, {5728.0}, {6704.0}, {7701.0}},
    {{949.0}, {1896.0}, {2865.0}, {3824.0}, {4787.0}, {5728.0}, {6704.0}, {7702.0}},
    {{949.0}, {1896.0}, {2865.0}, {3824.0}, {4788.0}, {5712.0}, {6691.0}, {7735.0}},
    {{938.0}, {1872.0}, {2817.0}, {3752.0}, {4717.0}, {5714.0}, {6729.0}, {7750.0}},
    {{940.0}, {1872.0}, {2824.0}, {3757.0}, {4694.0}, {5642.0}, {6671.0}, {7755.0}},
    {{972.0}, {1941.0}, {2904.0}, {3876.0}, {4853.0}, {5821.0}, {6778.0}, {7760.0}},
    {{934.0}, {1869.0}, {2818.0}, {3747.0}, {4693.0}, {5624.0}, {6651.0}, {7777.0}},
    {{944.0}, {1881.0}, {2822.0}, {3768.0}, {4740.0}, {5727.0}, {6756.0}, {7790.0}},
    {{936.0}, {1875.0}, {2819.0}, {3763.0}, {4736.0}, {5713.0}, {6741.0}, {7796.0}},
    {{934.0}, {1869.0}, {2817.0}, {3750.0}, {4700.0}, {5690.0}, {6733.0}, {7797.0}},
    {{951.0}, {1900.0}, {2862.0}, {3824.0}, {4788.0}, {5765.0}, {6768.0}, {7805.0}},
    {{945.0}, {1896.0}, {2865.0}, {3823.0}, {4786.0}, {5749.0}, {6766.0}, {7809.0}},
    {{933.0}, {1871.0}, {2818.0}, {3753.0}, {4720.0}, {5700.0}, {6740.0}, {7811.0}},
    {{938.0}, {1877.0}, {2824.0}, {3778.0}, {4760.0}, {5728.0}, {6733.0}, {7814.0}},
    {{965.0}, {1933.0}, {2903.0}, {3857.0}, {4820.0}, {5796.0}, {6813.0}, {7819.0}},
    {{968.0}, {1934.0}, {2905.0}, {3875.0}, {4849.0}, {5813.0}, {6802.0}, {7820.0}},
    {{947.0}, {1910.0}, {2872.0}, {3832.0}, {4787.0}, {5749.0}, {6761.0}, {7821.0}},
    {{934.0}, {1871.0}, {2821.0}, {3753.0}, {4714.0}, {5693.0}, {6802.0}, {7828.0}},
    {{972.0}, {1936.0}, {2908.0}, {3877.0}, {4852.0}, {5813.0}, {6802.0}, {7833.0}},
    {{970.0}, {1935.0}, {2906.0}, {3875.0}, {4851.0}, {5815.0}, {6816.0}, {7849.0}},
    {{949.0}, {1900.0}, {2865.0}, {3826.0}, {4808.0}, {5797.0}, {6821.0}, {7853.0}},
    {{934.0}, {1868.0}, {2815.0}, {3749.0}, {4695.0}, {5659.0}, {6698.0}, {7853.0}},
    {{936.0}, {1874.0}, {2822.0}, {3752.0}, {4698.0}, {5703.0}, {6739.0}, {7862.0}},
    {{969.0}, {1936.0}, {2908.0}, {3899.0}, {4904.0}, {5882.0}, {6854.0}, {7874.0}},
    {{968.0}, {1935.0}, {2908.0}, {3876.0}, {4851.0}, {5819.0}, {6846.0}, {7877.0}},
    {{973.0}, {1936.0}, {2906.0}, {3875.0}, {4852.0}, {5816.0}, {6817.0}, {7878.0}},
    {{973.0}, {1936.0}, {2908.0}, {3877.0}, {4852.0}, {5826.0}, {6843.0}, {7880.0}},
    {{974.0}, {1960.0}, {2941.0}, {3920.0}, {4900.0}, {5883.0}, {6875.0}, {7886.0}},
    {{947.0}, {1901.0}, {2872.0}, {3833.0}, {4826.0}, {5814.0}, {6844.0}, {7891.0}},
    {{937.0}, {1877.0}, {2824.0}, {3791.0}, {4785.0}, {5758.0}, {6815.0}, {7894.0}},
    {{939.0}, {1876.0}, {2822.0}, {3752.0}, {4719.0}, {5734.0}, {6826.0}, {7896.0}},
    {{953.0}, {1910.0}, {2871.0}, {3831.0}, {4786.0}, {5756.0}, {6779.0}, {7897.0}},
    {{950.0}, {1910.0}, {2871.0}, {3831.0}, {4787.0}, {5736.0}, {6804.0}, {7899.0}},
    {{932.0}, {1869.0}, {2815.0}, {3748.0}, {4694.0}, {5616.0}, {6592.0}, {7900.0}},
    {{948.0}, {1909.0}, {2873.0}, {3841.0}, {4831.0}, {5813.0}, {6846.0}, {7904.0}},
    {{933.0}, {1870.0}, {2815.0}, {3750.0}, {4719.0}, {5729.0}, {6810.0}, {7929.0}},
    {{942.0}, {1878.0}, {2824.0}, {3752.0}, {4698.0}, {5654.0}, {6727.0}, {7945.0}},
    {{934.0}, {1872.0}, {2820.0}, {3791.0}, {4796.0}, {5799.0}, {6868.0}, {7953.0}},
    {{950.0}, {1922.0}, {2907.0}, {3875.0}, {4853.0}, {5841.0}, {6886.0}, {7958.0}},
    {{983.0}, {1964.0}, {2944.0}, {3920.0}, {4899.0}, {5884.0}, {6906.0}, {7961.0}},
    {{984.0}, {1961.0}, {2942.0}, {3919.0}, {4911.0}, {5911.0}, {6929.0}, {7963.0}},
    {{935.0}, {1875.0}, {2822.0}, {3780.0}, {4773.0}, {5795.0}, {6868.0}, {7964.0}},
    {{952.0}, {1935.0}, {2909.0}, {3887.0}, {4891.0}, {5884.0}, {6919.0}, {7970.0}},
    {{941.0}, {1886.0}, {2863.0}, {3822.0}, {4809.0}, {5826.0}, {6896.0}, {7978.0}},
    {{980.0}, {1959.0}, {2941.0}, {3920.0}, {4900.0}, {5897.0}, {6929.0}, {7998.0}},
    {{970.0}, {1934.0}, {2906.0}, {3876.0}, {4852.0}, {5881.0}, {6929.0}, {7998.0}},
    {{934.0}, {1871.0}, {2818.0}, {3764.0}, {4772.0}, {5810.0}, {6911.0}, {7999.0}},
    {{1009.0}, {2000.0}, {3007.0}, {3995.0}, {5005.0}, {5979.0}, {6986.0}, {8021.0}},
    {{1007.0}, {1999.0}, {3007.0}, {3994.0}, {5005.0}, {5979.0}, {6987.0}, {8021.0}},
    {{949.0}, {1911.0}, {2882.0}, {3872.0}, {4879.0}, {5887.0}, {6938.0}, {8025.0}},
    {{1007.0}, {2005.0}, {3008.0}, {3996.0}, {5006.0}, {6001.0}, {7016.0}, {8033.0}},
    {{949.0}, {1911.0}, {2873.0}, {3836.0}, {4828.0}, {5828.0}, {6888.0}, {8056.0}},
    {{970.0}, {1933.0}, {2909.0}, {3887.0}, {4894.0}, {5917.0}, {6990.0}, {8062.0}},
    {{945.0}, {1896.0}, {2864.0}, {3824.0}, {4842.0}, {5887.0}, {6930.0}, {8063.0}},
    {{953.0}, {1901.0}, {2865.0}, {3826.0}, {4823.0}, {5837.0}, {6927.0}, {8070.0}},
    {{984.0}, {1961.0}, {2943.0}, {3920.0}, {4911.0}, {5922.0}, {6991.0}, {8074.0}},
    {{938.0}, {1874.0}, {2820.0}, {3761.0}, {4721.0}, {5696.0}, {6790.0}, {8110.0}},
    {{938.0}, {1876.0}, {2822.0}, {3824.0}, {4885.0}, {5979.0}, {7052.0}, {8113.0}},
    {{934.0}, {1870.0}, {2819.0}, {3757.0}, {4766.0}, {5805.0}, {6917.0}, {8118.0}},
    {{972.0}, {1935.0}, {2907.0}, {3875.0}, {4851.0}, {5826.0}, {6946.0}, {8119.0}},
    {{980.0}, {1959.0}, {2941.0}, {3920.0}, {4936.0}, {5976.0}, {7051.0}, {8125.0}},
    {{946.0}, {1911.0}, {2872.0}, {3832.0}, {4826.0}, {5837.0}, {6911.0}, {8137.0}},
    {{966.0}, {1934.0}, {2908.0}, {3894.0}, {4945.0}, {5987.0}, {7085.0}, {8151.0}},
    {{932.0}, {1871.0}, {2816.0}, {3749.0}, {4696.0}, {5706.0}, {6900.0}, {8151.0}},
    {{939.0}, {1876.0}, {2824.0}, {3800.0}, {4788.0}, {5841.0}, {6989.0}, {8152.0}},
    {{968.0}, {1935.0}, {2907.0}, {3876.0}, {4855.0}, {5894.0}, {7008.0}, {8169.0}},
    {{935.0}, {1872.0}, {2818.0}, {3751.0}, {4693.0}, {5671.0}, {6841.0}, {8192.0}},
    {{932.0}, {1869.0}, {2820.0}, {3802.0}, {4818.0}, {5854.0}, {6999.0}, {8214.0}},
    {{982.0}, {1962.0}, {2943.0}, {3927.0}, {4944.0}, {5957.0}, {7075.0}, {8237.0}},
    {{945.0}, {1896.0}, {2863.0}, {3841.0}, {4901.0}, {5973.0}, {7090.0}, {8249.0}},
    {{946.0}, {1922.0}, {2907.0}, {3902.0}, {4959.0}, {6019.0}, {7132.0}, {8268.0}},
    {{983.0}, {1963.0}, {2945.0}, {3942.0}, {4976.0}, {6012.0}, {7132.0}, {8275.0}},
    {{971.0}, {1934.0}, {2905.0}, {3886.0}, {4897.0}, {5943.0}, {7073.0}, {8278.0}},
    {{955.0}, {1935.0}, {2933.0}, {3952.0}, {4990.0}, {6064.0}, {7185.0}, {8284.0}},
    {{933.0}, {1870.0}, {2815.0}, {3748.0}, {4693.0}, {5596.0}, {6522.0}, {8285.0}},
    {{994.0}, {1986.0}, {2995.0}, {3994.0}, {5004.0}, {6031.0}, {7172.0}, {8303.0}},
    {{983.0}, {1962.0}, {2943.0}, {3920.0}, {4912.0}, {5940.0}, {7071.0}, {8304.0}},
    {{932.0}, {1870.0}, {2816.0}, {3770.0}, {4823.0}, {5983.0}, {7157.0}, {8308.0}},
    {{951.0}, {1910.0}, {2873.0}, {3867.0}, {4927.0}, {6006.0}, {7142.0}, {8322.0}},
    {{934.0}, {1876.0}, {2845.0}, {3835.0}, {4878.0}, {5947.0}, {7083.0}, {8345.0}},
    {{968.0}, {1934.0}, {2907.0}, {3877.0}, {4851.0}, {5833.0}, {7037.0}, {8352.0}},
    {{994.0}, {1986.0}, {2996.0}, {3995.0}, {5039.0}, {6088.0}, {7200.0}, {8361.0}},
    {{939.0}, {1879.0}, {2820.0}, {3795.0}, {4792.0}, {5896.0}, {7044.0}, {8365.0}},
    {{955.0}, {1921.0}, {2905.0}, {3877.0}, {4878.0}, {5885.0}, {6976.0}, {8370.0}},
    {{971.0}, {1960.0}, {2943.0}, {3921.0}, {4944.0}, {5991.0}, {7115.0}, {8380.0}},
    {{1002.0}, {1975.0}, {2982.0}, {3994.0}, {5005.0}, {6075.0}, {7229.0}, {8383.0}},
    {{950.0}, {1923.0}, {2910.0}, {3922.0}, {4998.0}, {6090.0}, {7219.0}, {8394.0}},
    {{934.0}, {1870.0}, {2816.0}, {3749.0}, {4738.0}, {5841.0}, {7074.0}, {8416.0}},
    {{935.0}, {1874.0}, {2910.0}, {4038.0}, {5331.0}, {6357.0}, {7394.0}, {8445.0}},
    {{996.0}, {1986.0}, {2996.0}, {3994.0}, {5004.0}, {6029.0}, {7154.0}, {8454.0}},
    {{941.0}, {1877.0}, {2824.0}, {3801.0}, {4833.0}, {5921.0}, {7142.0}, {8457.0}},
    {{1009.0}, {2016.0}, {3032.0}, {4037.0}, {5089.0}, {6171.0}, {7325.0}, {8490.0}},
    {{1007.0}, {2030.0}, {3079.0}, {4136.0}, {5219.0}, {6291.0}, {7393.0}, {8506.0}},
    {{1008.0}, {2031.0}, {3079.0}, {4129.0}, {5216.0}, {6293.0}, {7394.0}, {8512.0}},
    {{1010.0}, {2000.0}, {3007.0}, {4013.0}, {5064.0}, {6137.0}, {7293.0}, {8520.0}},
    {{1030.0}, {2072.0}, {3122.0}, {4161.0}, {5220.0}, {6292.0}, {7394.0}, {8533.0}},
    {{966.0}, {1935.0}, {2936.0}, {3953.0}, {5033.0}, {6186.0}, {7383.0}, {8540.0}},
    {{948.0}, {1911.0}, {2873.0}, {3878.0}, {4952.0}, {6048.0}, {7258.0}, {8545.0}},
    {{967.0}, {1937.0}, {2935.0}, {3944.0}, {5084.0}, {6222.0}, {7378.0}, {8563.0}},
    {{981.0}, {1960.0}, {2942.0}, {3919.0}, {4917.0}, {6002.0}, {7283.0}, {8569.0}},
    {{981.0}, {1972.0}, {2985.0}, {4008.0}, {5092.0}, {6216.0}, {7402.0}, {8590.0}},
    {{941.0}, {1901.0}, {2905.0}, {3946.0}, {5055.0}, {6191.0}, {7391.0}, {8611.0}},
    {{936.0}, {1869.0}, {2895.0}, {3953.0}, {5025.0}, {6156.0}, {7402.0}, {8810.0}},
    {{1007.0}, {2031.0}, {3079.0}, {4128.0}, {5215.0}, {6311.0}, {7550.0}, {8834.0}},
    {{969.0}, {1934.0}, {2906.0}, {3877.0}, {4937.0}, {6080.0}, {7375.0}, {8850.0}},
    {{966.0}, {1929.0}, {2904.0}, {3877.0}, {4917.0}, {6078.0}, {7409.0}, {8858.0}},
    {{1028.0}, {2072.0}, {3122.0}, {4172.0}, {5273.0}, {6405.0}, {7610.0}, {8882.0}},
    {{966.0}, {1933.0}, {2909.0}, {3907.0}, {4968.0}, {6155.0}, {7466.0}, {8895.0}},
    {{983.0}, {1962.0}, {2940.0}, {3918.0}, {4898.0}, {5926.0}, {7051.0}, {8896.0}},
    {{981.0}, {1962.0}, {2982.0}, {4028.0}, {5125.0}, {6269.0}, {7531.0}, {8904.0}},
    {{947.0}, {1935.0}, {2985.0}, {4039.0}, {5192.0}, {6439.0}, {7752.0}, {9148.0}},
    {{1078.0}, {2178.0}, {3268.0}, {4375.0}, {5527.0}, {6692.0}, {7945.0}, {9347.0}},
    {{1061.0}, {2213.0}, {3383.0}, {4534.0}, {5729.0}, {6913.0}, {8160.0}, {9419.0}}};
    long double epsilon = 5;
    unsigned long int min_points = 2;
    unsigned long int max_points = 4;
    density::fuzzy::CoreDBSCAN<double> clf = density::fuzzy::CoreDBSCAN<double>(epsilon, min_points, max_points, distance::euclidean);
    std::vector<std::map<int, double> > clusters = clf.predict(data);
    /*long double min_epsilon = 2.1;
    long double max_epsilon = 6;
    unsigned long int min_points = 1;
    density::fuzzy::BorderDBSCAN<long double> clf = density::fuzzy::BorderDBSCAN<long double>(min_epsilon, max_epsilon, min_points, euclidean);
    std::vector<std::unordered_map<int, long double> > clusters = clf.predict(other_data);
    for (auto i = clusters.begin(); i != clusters.end(); ++i) {
        size_t index = std:vg:distance(clusters.begin(), i);
        std::cout << "Index: " << index << ", Point: " << other_data.at(index).at(0) << "    ";
        print_map(*i);
        std::cout << '\n';
    }*/

    density::DBPack2<double> dbpack2 = density::DBPack2<double>(5.0, 3);
    std::vector<int> single_clusters = dbpack2.predict(single_data);
    for (size_t i = 0; i < single_clusters.size(); ++i) {
        std::cout << "Single Clusters: Row #" << i << " - "  << single_data.at(i) << " : Cluster #" << single_clusters.at(i) << std::endl;
    }

    long double min_epsilon = 1;
    long double max_epsilon = 10;
    min_points = 2;
    max_points = 4;
    density::fuzzy::DBSCAN<double> fuzzy_clf = density::fuzzy::DBSCAN<double>(min_epsilon, max_epsilon, min_points, max_points, distance::euclidean<double>);
    clusters = fuzzy_clf.predict(other_data);
    for (auto i = clusters.begin(); i != clusters.end(); ++i) {
        size_t index = std::distance(clusters.begin(), i);
        std::cout << "Index: " << index << ", Point: " << other_data.at(index).at(0) << " - Memberships: ";
        print_map(*i);
        std::cout << '\n';
    }
    double similarity = similarity::fuzzy::hwang_yang_hung<int>(clusters, 6, 7);
    std::cout << "similarity: " << similarity << "\n";
    epsilon = 5.0;
    min_points = 2;
    density::DBSCAN<double> db_clf = density::DBSCAN<double>(epsilon, min_points, distance::euclidean<double>);
    density::moving::MovingDBSCAN<double> moving_clf = density::moving::MovingDBSCAN<double>(db_clf, 0.5);
    std::vector<std::vector<int> > results = moving_clf.predict(sequential_data);
    for (size_t i = 0; i < results.size(); i++) {
        std::cout << "\n Row #" << i << ": ";
        for (size_t j = 0; j < results[i].size(); j++) {
            std::cout << results[i][j] << "(" << sequential_data[i][j][0] << "), ";
        }
    }
    unsigned int k = 4;
    unsigned int m = 5;
    density::moving::CMC<double> convoy_clf = density::moving::CMC<double>(db_clf, k, m);
    std::vector<std::vector<size_t> > convoy_indices;
    std::vector<size_t> start_times, end_times;
    std::tie(convoy_indices, start_times, end_times) = convoy_clf.predict(sequential_data);
    std::cout << "Size: " << convoy_indices.size() << "\n";
    for (size_t i = 0; i < start_times.size(); ++i) {
        print_vector<size_t>(convoy_indices[i]);
        std::cout << "Start: " << start_times[i] << ", End: " << end_times[i] << "\n";
    }

    long int kmeans_k = 5;
    long int max_iterations = 100;
    double tolerance = 1;
    std::vector<std::vector<double >> centroids;
    std::vector<long int> kmeans_clusters;
    clustering::KMeans<double> kmeans_clf = clustering::KMeans<double>(kmeans_k, max_iterations, tolerance, distance::euclidean<double>);
    std::tie(centroids, kmeans_clusters) = kmeans_clf.predict(other_data);
    print_vector(kmeans_clusters);

    std::vector<long int> kmedian_clusters;
    clustering::KMedian<double> kmedian_clf = clustering::KMedian<double>(kmeans_k, max_iterations, tolerance, distance::euclidean<double>);
    std::tie(centroids, kmedian_clusters) = kmedian_clf.predict(other_data);
    print_vector(kmedian_clusters);

    clustering::KMode<long int> kmode_clf = clustering::KMode<long int>(kmeans_k, max_iterations, tolerance, distance::euclidean<long int>);

    density::fuzzy::BorderDBPack<double, long int> pack_clf = density::fuzzy::BorderDBPack<double, long int>(2.0, 7.0, 2);
    std::vector<std::map<long int, double> > pack_clusters = pack_clf.predict(single_data);
    for (auto i = pack_clusters.begin(); i != pack_clusters.end(); ++i) {
        size_t index = std::distance(pack_clusters.begin(), i);
        std::cout << "Index: " << index << ", Point: " << single_data.at(index) << "    ";
        print_map(*i);
        std::cout << '\n';
    }

    return 0;
}
