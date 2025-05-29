
class KMeans {
    constructor(k, iterations, tolerance, distance) {
        this.k = k;
        this.iterations = iterations;
        this.tolerance = tolerance;
        this.distance = distance;
    }

    async predict(data) {
        var dimensions = data[0].length,
            transformedData = this.constructor.flatten(data);
        
            if (window.crossOriginIsolated) {
                // Create a SharedArrayBuffer to hold the imageData
                sharedBuffer = new SharedArrayBuffer(transformedData.length),
                sharedData = new Float32Array(sharedBuffer);

                // Copy the imageData to the SharedArrayBuffer
                sharedData.set(transformedData.data);
                transformedData = sharedData;
            }

        var module = await new HIGHP();
        var clf = new module.KMeans(this.k, this.iterations, this.tolerance, dimensions, this.distance);
        return clf.predict(transformedData);
    }

    static flatten(data) {
        var dimensions = data[0].length,
            transformedData = new Float32Array(data.length * dimensions);
        // flatten data for classification
        for (var i = 0; i < data.length; i++) {
            var dataEntry = data[i];
            for (var j = 0; j < dimensions; j++) {
                transformedData[i * dimensions + j] = dataEntry[j];
            }
        }
        return transformedData;
    }
}