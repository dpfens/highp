CC=gcc
CCFLAGS = -Wall -fopenmp -g -O2
CPP=g++
CPPFLAGS = -Wall -fopenmp -g -O2 -std=c++11
SWIG=./swig/
SRC=./src/
SRC_C=$(SRC)c/
SRC_CPP=$(SRC)cpp/
BIN=./bin/
PYTHONLIB=highp
PYTHON27CFLAGS = $(shell python-config --cflags)
PYTHON27INCLUDES = $(shell python-config --includes)
PYTHON27LDFLAGS = $(shell python-config --ldflags)
PYTHON3CFLAGS = $(shell python3-config --cflags)
PYTHON3INCLUDES = $(shell python3-config --includes)
PYTHON3LDFLAGS = $(shell python3-config --ldflags)

test:
	mkdir -p $(BIN)
	$(CC) $(CCFLAGS) -o $(BIN)test $(SRC_C)test.c -lm
	# $(BIN)test
	$(CPP) $(CPPFLAGS) -o $(BIN)test $(SRC_CPP)test.cpp
	$(BIN)test

python:
	python setup.py build_ext --inplace

build:
	mkdir -p $(PYTHONLIB)
	echo "" > $(PYTHONLIB)/__init__.py
	swig -c++ -python -outdir $(PYTHONLIB) $(SWIG)fuzzy.i
	swig -c++ -python -outdir $(PYTHONLIB) $(SWIG)distance.i
	swig -c++ -python -outdir $(PYTHONLIB) $(SWIG)dbscan.i
	swig -c++ -python -outdir $(PYTHONLIB) $(SWIG)moving.i
	swig -c++ -python -outdir $(PYTHONLIB) $(SWIG)similarity.i
	swig -c++ -python -outdir $(PYTHONLIB) $(SWIG)kmeans.i
	python setup.py build_ext --inplace

install:
	mkdir -p $(PYTHONLIB)
	echo "" > $(PYTHONLIB)/__init__.py
	swig -c++ -python -outdir $(PYTHONLIB) $(SWIG)fuzzy.i
	swig -c++ -python -outdir $(PYTHONLIB) $(SWIG)distance.i
	swig -c++ -python -outdir $(PYTHONLIB) $(SWIG)dbscan.i
	swig -c++ -python -outdir $(PYTHONLIB) $(SWIG)moving.i
	swig -c++ -python -outdir $(PYTHONLIB) $(SWIG)similarity.i
	swig -c++ -python -outdir $(PYTHONLIB) $(SWIG)kmeans.i
	python setup.py install

clean:
	rm -rf build dist bin swig/*.cxx $(PYTHONLIB)
