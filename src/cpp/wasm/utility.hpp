#ifndef WASM_UTILITY_H
#define WASM_UTILITY_H

#include <emscripten/val.h>

namespace wasm {

    namespace utility {

        template <typename T>
        emscripten::val contiguousVecToArray(T* data, long int dataLength) {
            emscripten::val arr = emscripten::val::array();
            size_t i = 0;
            for (i = 0; i < dataLength; ++i) {
                arr.call<void>("push", std::move(data[i]));
            }
            return arr;
        }

        template <typename T>
        std::vector<T> arrayToVec(emscripten::val array) {
            if (!array.isArray()) {
                throw std::runtime_error("Input is not a valid array");
            }
            unsigned int length = array["length"].as<unsigned int>();
            std::vector<T> result = std::vector<T>(length);
            for (unsigned int i = 0; i < length; ++i) {
                result[i] = array[i].as<T>();
            }
            return result;
        }

        template <typename T>
        std::vector<std::vector<T>> array2DToVec(emscripten::val array) {
            if (!array.isArray()) {
                throw std::runtime_error("Input is not a valid array");
            }
            // Dimensions
            unsigned int arrayLength = array["length"].as<unsigned int>();
            unsigned int columnCount = array[0]["length"].as<unsigned int>(); 
            bool hasBuffer = array[0].hasOwnProperty("buffer");

            // Pre-allocate result vector
            std::vector<std::vector<double>> result(arrayLength);

            // Optimize assuming contiguous inner arrays:
            for (unsigned int i = 0; i < arrayLength; ++i) {
                emscripten::val innerArray = array[i];

                // Get a direct pointer (if possible)
                T* rowData = nullptr;
                if (hasBuffer) { 
                    emscripten::val buffer = innerArray["buffer"]; 
                    rowData = reinterpret_cast<T*>(buffer.as<uintptr_t>());
                }

                if (rowData) {
                    // Super-efficient copy from contiguous data
                    result[i].assign(rowData, rowData + columnCount); 
                } else {
                    // Fallback to the original method if no contiguous buffer
                    result[i].reserve(columnCount);
                    for (size_t col = 0; col < columnCount; ++col) {
                        result[i].push_back(innerArray[col].as<T>());
                    }
                }
            }
            return result;
        }

        template <typename T>
        emscripten::val vecToArray(const std::vector<T>& data) {
            emscripten::val arr = emscripten::val::array();
            for (const auto& value : data) {
                arr.call<void>("push", std::move(value));
            }
            return arr;
        }


        template <class T1, class T2>
        emscripten::val mapToObject(const std::map<T1, T2>& cppMap) {
            emscripten::val jsObj = emscripten::val::object();
            for (const auto& pair : cppMap) {
                jsObj.set(std::to_string(pair.first), pair.second);
            }
            return jsObj;
        }

                template <typename T>
        emscripten::val vecToTypedArray(const std::vector<T>& data) {
            size_t dataSize = data.size() * sizeof(T);

            // Allocate an ArrayBuffer
            emscripten::val arrayBuffer = emscripten::val::global("ArrayBuffer").new_(dataSize);

            // Access the ArrayBuffer's data directly
            void* arrayBufferMemory = arrayBuffer["data"].as<void*>();

            // Copy the data into the ArrayBuffer's memory
            std::memcpy(arrayBufferMemory, data.data(), dataSize);

            // Create a Float32Array view on the ArrayBuffer
            emscripten::val float32Array = emscripten::val::global("Float32Array").new_(arrayBuffer);

            return float32Array;
        }

    }
}

#endif /* WASM_UTILITY_H */