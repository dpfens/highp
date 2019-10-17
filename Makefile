CC=gcc
CCFLAGS = -Wall -fopenmp -g
SRC=./src/c/
BIN=./bin/
PYTHON27CFLAGS = $(shell python-config --cflags)
PYTHON27INCLUDES = $(shell python-config --includes)
PYTHON27LDFLAGS = $(shell python-config --ldflags)
PYTHON3CFLAGS = $(shell python3-config --cflags)
PYTHON3INCLUDES = $(shell python3-config --includes)
PYTHON3LDFLAGS = $(shell python3-config --ldflags)

test:
	mkdir -p $(BIN)
	$(CC) $(CCFLAGS) -o $(BIN)test $(SRC)test.c -lm

python:
	python setup.py build_ext --inplace

build:
	mkdir -p $(BIN)
	swig -outdir $(BIN) -python $(SRC)dbscan.i 

	$(CC) -fPIC -c $(SRC)dbscan.c $(SRC)dbscan_wrap.c $(PYTHON27INCLUDES)

	ld -shared dbscan_wrap.o dbscan.o -o $(BIN)_cluster.so
