#ifndef WASM_UTILITY_H
#define WASM_UTILITY_H

#include <emscripten/val.h>

namespace wasm {

    namespace utility {

        template <typename T>
        emscripten::val vecToArray(const std::vector<T>& data) {
            emscripten::val arr = emscripten::val::array();
            for (const auto& value : data) {
                arr.call<void>("push", value);
            }
            return arr;
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