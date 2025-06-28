/**
 * High-performance K-Means clustering implementation using WebAssembly.
 * Provides efficient clustering with optimized memory management for large datasets.
 * 
 * @class KMeans
 * @example
 * // Basic usage
 * const kmeans = new KMeans(3, 100, 0.001);
 * const result = await kmeans.predict(data);
 * kmeans.destroy(); // Important: clean up resources
 * 
 * @example
 * // Advanced usage with error handling
 * const kmeans = new KMeans(5, 200, 0.0001);
 * try {
 *   await kmeans.init();
 *   const result = await kmeans.predict(largeDataset);
 *   console.log('Clusters:', result.clusters);
 *   console.log('Centroids:', result.centroids);
 * } catch (error) {
 *   console.error('Clustering failed:', error);
 * } finally {
 *   kmeans.destroy();
 * }
 */
class KMeans {
    /**
     * Creates a new KMeans clustering instance.
     * 
     * @param {number} k - Number of clusters to create (must be > 0)
     * @param {number} iterations - Maximum number of iterations for convergence (default: 100)
     * @param {number} tolerance - Convergence tolerance threshold (default: 0.001)
     * @throws {Error} If k is not a positive integer
     */
    constructor(k, iterations, tolerance) {
        if (!Number.isInteger(k) || k <= 0) {
            throw new Error('k must be a positive integer');
        }
        
        /** @type {number} Number of clusters */
        this.k = k;
        
        /** @type {number} Maximum iterations for convergence */
        this.iterations = iterations || 100;
        
        /** @type {number} Convergence tolerance threshold */
        this.tolerance = tolerance || 0.001;
        
        /** @type {Object|null} WebAssembly module instance */
        this.module = null;
        
        /** @type {Object|null} C++ KMeans classifier instance */
        this.clf = null;
        
        /** @type {WasmBufferManager|null} Memory buffer manager for efficient data transfer */
        this.bufferManager = null;
    }

    /**
     * Initializes the WebAssembly module and buffer manager.
     * Must be called before using predict() method.
     * 
     * @async
     * @returns {Promise<Object>} The initialized WASM module
     * @throws {Error} If WebAssembly module fails to load
     */
    async init() {
        if (!this.module) {
            try {
                this.module = await new HIGHP();
                this.bufferManager = new WasmBufferManager(this.module);
            } catch (error) {
                throw new Error(`Failed to initialize WebAssembly module: ${error.message}`);
            }
        }
        return this.module;
    }

    /**
     * Performs K-means clustering on the provided dataset.
     * Uses optimized memory management for handling large datasets efficiently.
     * 
     * @async
     * @param {number[][]} data - Array of data points, where each point is an array of numbers
     * @returns {Promise<Object>} Clustering result containing clusters and centroids
     * @returns {number[]} result.clusters - Cluster assignments for each data point
     * @returns {number[][]} result.centroids - Final cluster centroids
     * @throws {Error} If data is invalid or clustering fails
     * 
     * @example
     * const data = [[1, 2], [2, 3], [8, 9], [9, 10]];
     * const result = await kmeans.predict(data);
     * // result.clusters: [0, 0, 1, 1]
     * // result.centroids: [[1.5, 2.5], [8.5, 9.5]]
     */
    async predict(data) {
        await this.init();
        
        // Validate input data structure and dimensions
        const validation = this.constructor.validateData(data);
        const dimensions = validation.dimensions;
        
        // Convert 2D array to flat Float64Array for WASM compatibility
        const transformedData = this.constructor.flatten(data);
        
        try {
            // Create KMeans instance if not exists (lazy initialization)
            if (!this.clf) {
                this.clf = new this.module.KMeans(
                    this.k, 
                    this.iterations, 
                    this.tolerance, 
                    dimensions
                );
            }

            // Use direct memory access for maximum performance
            // This bypasses JavaScript-WASM boundary overhead
            return this.bufferManager.processWithDirectAccess(
                transformedData, 
                (ptr, length) => {
                    return this.clf.predict(ptr, length);
                }
            );
            
        } catch (error) {
            console.error('KMeans prediction failed:', error);
            console.error('Data info:', {
                originalLength: data.length,
                dimensions: dimensions,
                transformedLength: transformedData.length,
                expectedLength: data.length * dimensions,
                arrayType: transformedData.constructor.name,
                bytesPerElement: transformedData.BYTES_PER_ELEMENT
            });
            throw new Error(`Clustering prediction failed: ${error.message}`);
        }
    }

    /**
     * Releases all allocated resources including WASM memory buffers.
     * Should always be called when the KMeans instance is no longer needed
     * to prevent memory leaks.
     * 
     * @example
     * const kmeans = new KMeans(3, 100, 0.001);
     * try {
     *   const result = await kmeans.predict(data);
     *   // use result...
     * } finally {
     *   kmeans.destroy(); // Always clean up
     * }
     */
    destroy() {
        if (this.bufferManager) {
            this.bufferManager.destroy();
            this.bufferManager = null;
        }
        if (this.clf) {
            // Note: Embind objects are garbage collected automatically
            // but we null the reference to help GC
            this.clf = null;
        }
        this.module = null;
    }

    /**
     * Validates the structure and integrity of input data.
     * Ensures data is a non-empty 2D array with consistent dimensions.
     * 
     * @static
     * @param {number[][]} data - Input data to validate
     * @returns {Object} Validation result with data dimensions
     * @returns {number} result.points - Number of data points
     * @returns {number} result.dimensions - Number of dimensions per point
     * @throws {Error} If data is invalid or inconsistent
     */
    static validateData(data) {
        if (!Array.isArray(data) || data.length === 0) {
            throw new Error('Data must be a non-empty array');
        }
        
        if (!Array.isArray(data[0])) {
            throw new Error('Data must be a 2D array (array of arrays)');
        }
        
        const dimensions = data[0].length;
        if (dimensions === 0) {
            throw new Error('Data points must have at least one dimension');
        }
        
        // Validate that all points have the same number of dimensions
        for (let i = 1; i < data.length; i++) {
            if (!Array.isArray(data[i]) || data[i].length !== dimensions) {
                throw new Error(`Inconsistent dimensions at point ${i}: expected ${dimensions}, got ${data[i]?.length || 'undefined'}`);
            }
        }
        
        return { points: data.length, dimensions };
    }

    /**
     * Converts 2D data array to a flat Float64Array for efficient WASM transfer.
     * Uses row-major order: [point0_dim0, point0_dim1, ..., point1_dim0, point1_dim1, ...]
     * 
     * @static
     * @param {number[][]} data - 2D array of data points
     * @returns {Float64Array} Flattened array compatible with C++ double* arrays
     * 
     * @example
     * const data = [[1, 2], [3, 4]];
     * const flat = KMeans.flatten(data);
     * // Result: Float64Array [1, 2, 3, 4]
     */
    static flatten(data) {
        const dimensions = data[0].length;
        // Use Float64Array to match C++ double* expectation for precision
        const transformedData = new Float64Array(data.length * dimensions);
        
        // Flatten data using row-major order for efficient memory access
        for (let i = 0; i < data.length; i++) {
            const dataEntry = data[i];
            for (let j = 0; j < dimensions; j++) {
                transformedData[i * dimensions + j] = dataEntry[j];
            }
        }
        return transformedData;
    }
}

/**
 * Efficient memory buffer manager for WebAssembly operations.
 * Handles allocation, reuse, and cleanup of WASM heap memory to minimize
 * garbage collection and improve performance for repeated operations.
 * 
 * Memory Management Strategy:
 * 1. Allocates buffers on the WASM heap using _malloc()
 * 2. Reuses existing buffers when possible to avoid allocation overhead
 * 3. Grows buffers automatically when more space is needed
 * 4. Provides direct memory access to avoid JS-WASM boundary costs
 * 5. Properly cleans up all allocated memory on destruction
 * 
 * @class WasmBufferManager
 */
class WasmBufferManager {
    /**
     * Creates a new buffer manager for the given WASM module.
     * 
     * @param {Object} module - Initialized WebAssembly module with memory management functions
     */
    constructor(module) {
        /** @type {Object} WebAssembly module instance */
        this.module = module;
        
        /** @type {number|null} Pointer to allocated buffer in WASM heap */
        this.buffer = null;
        
        /** @type {number} Current buffer size in bytes */
        this.bufferSize = 0;
    }

    /**
     * Gets a buffer of at least the required size, allocating or growing as needed.
     * 
     * Memory Allocation Strategy:
     * - Reuses existing buffer if it's large enough
     * - Allocates new buffer with 20% extra space for future growth
     * - Frees old buffer before allocating new one to prevent leaks
     * - Uses WASM _malloc() for direct heap allocation
     * 
     * @param {number} requiredBytes - Minimum number of bytes needed
     * @returns {number} Pointer to the buffer in WASM memory space
     */
    getBuffer(requiredBytes) {
        // Check if current buffer is sufficient
        if (this.bufferSize < requiredBytes) {
            // Free old buffer first to prevent memory leaks
            if (this.buffer) {
                this.module._free(this.buffer);
            }
            
            // Allocate new buffer with extra space to reduce future allocations
            // 20% extra space balances memory usage vs allocation frequency
            const allocSize = Math.max(requiredBytes, Math.ceil(requiredBytes * 1.2));
            this.buffer = this.module._malloc(allocSize);
            this.bufferSize = allocSize;
            
            console.log(`Allocated ${allocSize} bytes for KMeans data (requested: ${requiredBytes})`);
        }
        
        return this.buffer;
    }

    /**
     * Processes typed array data using direct WASM memory access for optimal performance.
     * 
     * Direct Memory Access Process:
     * 1. Validates input is a supported typed array (Float32Array or Float64Array)
     * 2. Calculates required memory size based on array type and length
     * 3. Obtains/allocates appropriate buffer in WASM heap
     * 4. Maps buffer pointer to correct WASM heap view (HEAPF32 or HEAPF64)
     * 5. Copies JavaScript typed array directly to WASM memory
     * 6. Calls processing function with raw pointer (no JS-WASM conversion overhead)
     * 7. Returns result without copying data back (processing function handles this)
     * 
     * @param {Float32Array|Float64Array} typedArray - Input data array
     * @param {Function} processingFunction - Function to call with (pointer, length)
     * @returns {*} Result from the processing function
     * @throws {Error} If array type is not supported
     * 
     * @example
     * const data = new Float64Array([1.0, 2.0, 3.0, 4.0]);
     * const result = manager.processWithDirectAccess(data, (ptr, len) => {
     *   return wasmModule.someFunction(ptr, len);
     * });
     */
    processWithDirectAccess(typedArray, processingFunction) {
        // Validate input array type - only Float32/64Array supported for direct access
        if (!(typedArray instanceof Float32Array) && !(typedArray instanceof Float64Array)) {
            throw new Error('Direct access requires Float32Array or Float64Array');
        }

        // Calculate memory requirements
        const elementSize = typedArray.BYTES_PER_ELEMENT; // 4 bytes for Float32, 8 for Float64
        const byteSize = typedArray.length * elementSize;
        
        // Get buffer and calculate element-aligned index
        // WASM heap views are indexed by elements, not bytes
        const ptr = this.getBuffer(byteSize);
        const elementIndex = ptr / elementSize;
        
        // Select appropriate WASM heap view based on array precision
        // This ensures proper alignment and data interpretation
        let heapView;
        if (typedArray instanceof Float64Array) {
            heapView = this.module.HEAPF64;  // 8-byte double precision view
        } else if (typedArray instanceof Float32Array) {
            heapView = this.module.HEAPF32;  // 4-byte single precision view
        } else {
            throw new Error(`Unsupported typed array type: ${typedArray.constructor.name}`);
        }
        
        // Direct memory copy: JavaScript typed array → WASM heap
        // This is much faster than element-by-element copying
        heapView.set(typedArray, elementIndex);
        
        // Call processing function with raw pointer
        // The C++ code receives a direct pointer to the data
        return processingFunction(ptr, typedArray.length);
    }

    /**
     * Releases all allocated memory and resets the buffer manager.
     * Must be called to prevent memory leaks in the WASM heap.
     * 
     * Memory Cleanup Process:
     * 1. Calls WASM _free() to release heap memory
     * 2. Nulls buffer pointer to prevent double-free
     * 3. Resets size counter for fresh start
     */
    destroy() {
        if (this.buffer) {
            // Free WASM heap memory - critical for preventing memory leaks
            this.module._free(this.buffer);
            this.buffer = null;
            this.bufferSize = 0;
        }
    }
}