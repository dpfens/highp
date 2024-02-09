#ifndef WASM_UTILITY_H
#define WASM_UTILITY_H

#include <emscripten/val.h>

namespace wasm {

    namespace utility {

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
            unsigned int arrayLength = array["length"].as<unsigned int>();
            std::vector<std::vector<T>> result = std::vector<std::vector<T>>(arrayLength);
            for (unsigned int i = 0; i < arrayLength; ++i) {
                emscripten::val innerArray = array[i];

                if (!innerArray.isArray()) {
                    throw std::runtime_error("Inner elements must be arrays");
                }
                std::vector<T> innerVector;
                for (unsigned int j = 0; j < innerArray["length"].as<unsigned int>(); ++j) {
                    innerVector.push_back(innerArray[j].as<T>());
                }
                result[i] = innerVector;
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