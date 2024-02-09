# HighP
C/C++ implementations of various clustering algorithms, trajectory processing, and binary similarity metrics with Python SWIG/WebAssembly bindings to select algorithms.

## Usage

### C++
C++ has broader access to more algorithmd including implementations for:
*  Dozens of Binary attribute similarity metrics
*  Trajectories
  *  Convoy identification
  *  Trajectory Stop detection
*  Maximum Subarray
*  KDTree
*  etc.

The examples of C++ usage can be found in `src/cpp/test.cpp`

#### Test
To build the C++ implementations and execute the unit tests, use the `make test` command:

```bash
make test
```

### Python
The Python bindings open a subset of the algorithms implemented in C++. The best examples for using the `python` binding are through the `test.py` module. 

#### Dependencies
Install `swig` to build the Python bindings locally:
```bash
sudo apt install swig
make python_build  # build in-place
make python_install # install library and bindings
```

### WebAssembly
The Python bindings open a subset of the algorithms implemented in C++. The best examples for using the WebAssembly are through the `webassembly.html` file.  To build and install the webAssembly bindings, use the `make wasm_prod` command:

```bash
make wasm_prod
```

Then open the `webassembly.html` file to see the output from the `wasm` module.