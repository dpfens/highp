from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension("clustering", ["src/cython/dbscan.pyx"], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
        Extension("clustering", ["cython/dbscan.c"], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    ]

setup(
    name='hpclustering',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
