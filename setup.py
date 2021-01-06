from distutils.core import setup
from distutils.extension import Extension

compile_args = ['--std=c++11', '-fopenmp']
link_args=['-lgomp']
dbscan_module = Extension('highp._dbscan',
                          sources=['swig/dbscan_wrap.cxx','src/cpp/dbscan.cpp'],
                          extra_compile_args=compile_args,
                          extra_link_args=link_args
                         )
moving_module = Extension('highp._moving',
                          sources=['swig/moving_wrap.cxx','src/cpp/moving.cpp'],
                          extra_compile_args=compile_args,
                          extra_link_args=link_args
                         )
kmeans_module = Extension('highp._kmeans',
                         sources=['swig/kmeans_wrap.cxx','src/cpp/kmeans.cpp'],
                         extra_compile_args=compile_args,
                         extra_link_args=link_args
                         )
fuzzy_module = Extension('highp._fuzzy',
                          sources=['swig/fuzzy_wrap.cxx','src/cpp/fuzzy.cpp'],
                          extra_compile_args=compile_args,
                          extra_link_args=link_args
                         )
distance_module = Extension('highp._distance',
                         sources=['swig/distance_wrap.cxx','src/cpp/distance.cpp'],
                         extra_compile_args=compile_args,
                         extra_link_args=link_args
                         )
similarity_module = Extension('highp._similarity',
                         sources=['swig/similarity_wrap.cxx','src/cpp/similarity.cpp'],
                         extra_compile_args=compile_args,
                         extra_link_args=link_args
                         )

setup (name = 'highp',
       version = '0.1',
       author      = "Doug Fenstermacher",
       description = """High performance implementations of various algorithms""",
       ext_modules = [dbscan_module, moving_module, kmeans_module, distance_module, fuzzy_module, similarity_module],
       py_modules = ["highp.fuzzy", "highp.distance", "highp.dbscan", "highp.moving", "highp.similarity"],
       )
