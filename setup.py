from distutils.core import setup
from distutils.extension import Extension

dbscan_module = Extension('highp._dbscan',
                          sources=['swig/dbscan_wrap.cxx','src/cpp/dbscan.cpp'],
                          extra_compile_args=['--std=c++11']
                         )
fuzzy_module = Extension('highp._fuzzy',
                          sources=['swig/fuzzy_wrap.cxx','src/cpp/fuzzy.cpp'],
                          extra_compile_args=['--std=c++11']
                         )
distance_module = Extension('highp._distance',
                         sources=['swig/distance_wrap.cxx','src/cpp/distance.cpp'],
                         extra_compile_args=['--std=c++11']
                         )

setup (name = 'highp',
       version = '0.1',
       author      = "Doug Fenstermacher",
       description = """High performance implementations of various algorithms""",
       ext_modules = [dbscan_module, distance_module, fuzzy_module],
       py_modules = ["highp.fuzzy", "highp.distance", "highp.dbscan"],
       )
