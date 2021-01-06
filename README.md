# HighP
C/C++ implementations of various clustering algorithms, trajectory processing, and binary similarity metrics with Python SWIG bindings to select algorithms.

## Usage

### Dependencies
Install `swig`:
```bash
sudo apt install swig
```

### Build
To build the Python bindings in-place, use the `make build` command:

```bash
make build
```

### Test
To build the C++, use the `make test` command:

```bash
make test
```

The examples of C++ usage can be found in `src/cpp/test.cpp`

### Python
The Python bindings open a subset of the algorithms implemented in C++. The best examples for using the `python` binding are through the `test.py` module.  To build and install the Python bindings, use the `make install` command:

```bash
make install
```

### C++
The C++ library includes implementations of dozens of algorithms:
*  Binary similarity metrics
*  Convoy identification
*  Trajectory Stop detection
*  etc.
