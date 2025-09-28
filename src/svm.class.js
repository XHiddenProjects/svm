class SVM {
    /**
     * Constructor initializes the Support Vector Machine with default or user-specified options
     * @param {{cost: Number, tol: Number, maxPasses: Number, maxIterations: Number, kernel: 'linear'|'poly'|'rbf'|'sigmoid', 'alphaTol':Number, random: Math.random, kernelOptions: {degree: number, gamma: number, coef0: number, sigma: number}}} options Configuration options for the SVM
     */
    constructor(options = {}) {
        // Default options
        this.options = {
            cost: 1,
            tol: 1e-4,
            alphaTol: 1e-6,
            maxPasses: 10,
            maxIterations: 10000,
            kernel: 'linear', // 'linear', 'polynomial', 'rbf', 'sigmoid'
            kernelOptions: {}, // e.g., {degree: 3, gamma: 0.5, coef0: 1}
            whitening: true,
            random: Math.random
        };
        Object.assign(this.options, options);
        this._trained = false; // Indicates if the model has been trained
        this._loaded = false; // Indicates if the model has been loaded from JSON
        this.W = null; // For linear kernel weights
        this.minMax = null; // For whitening normalization parameters
        this.b = 0; // Bias term
        this.alphas = []; // Lagrange multipliers
        this.X = []; // Training features
        this.Y = []; // Original training labels
        this.labelsMap = {}; // Map from original labels to +1/-1
        this.inverseLabelsMap = {}; // Map back from +1/-1 to original labels
        this.supportVectorIdx = []; // Indices of support vectors
        this.confusion = {
            TP: 0,
            FP: 0,
            TN: 0,
            FN: 0
        }
        this.metrics = {
            accuracy: 0,
            F1: 0,
            recall: 0,
            precision: 0
        }
    }

    /**
     * Internal method to encode labels to +1/-1
     * @param {Array} labels - Original labels
     */
    _encodeLabels(labels) {
        const uniqueLabels = Array.from(new Set(labels));
        if (uniqueLabels.length !== 2) {
            throw new Error('This implementation supports binary classification with exactly two classes.');
        }
        this.labelsMap = {};
        this.inverseLabelsMap = {};
        this.labelsMap[uniqueLabels[0]] = -1;
        this.labelsMap[uniqueLabels[1]] = 1;
        this.inverseLabelsMap[-1] = uniqueLabels[0];
        this.inverseLabelsMap[1] = uniqueLabels[1];
        return labels.map(l => this.labelsMap[l]);
    }

    /**
     * Decodes internal +1/-1 labels back to original labels
     * @param {Number} label - Internal label (+1 or -1)
     * @returns {Any} Original label
     */
    _decodeLabel(label) {
        return this.inverseLabelsMap[label];
    }

    /**
     * Computes the confusion matrix components: TP, FP, TN, FN
     * @param {Array} features - Array of feature vectors
     * @param {Array} labels - True labels
     * @returns {SVM} Object with counts
     */
    _confusion(features, labels) {
        if (!this._trained && !this._loaded) {
            throw new Error('Model not trained or loaded');
        }
        if (features.length !== labels.length) {
            throw new Error('Features and labels length mismatch');
        }

        let TP = 0, FP = 0, TN = 0, FN = 0;

        for (let i = 0; i < features.length; i++) {
            const pred = this.predictOne(features[i]);
            const trueLabel = labels[i];

            if (pred === 1 && trueLabel === 1) {
                TP++;
            } else if (pred === 1 && trueLabel === -1) {
                FP++;
            } else if (pred === -1 && trueLabel === -1) {
                TN++;
            } else if (pred === -1 && trueLabel === 1) {
                FN++;
            }
        }
        this.confusion['TP'] = TP;
        this.confusion['FP'] = FP;
        this.confusion['TN'] = TN;
        this.confusion['FN'] = FN;
        return this;
    }

    /**
     * Calculates the accuracy of the model on given data
     * @param {Array} features - Array of feature vectors
     * @param {Array} labels - True labels
     * @returns {SVM} Accuracy as a proportion (0 to 1)
     */
    _accuracy(features, labels) {
        const predictions = this.predict(features);
        let correct = 0;
        for (let i = 0; i < labels.length; i++) {
            if (predictions[i] === labels[i]) {
                correct++;
            }
        }
        this.metrics['accuracy'] = correct / labels.length;
        return this;
    }

    /**
     * Calculates Precision (Positive Predictive Value)
     * @param {Array} features - Array of feature vectors
     * @param {Array} labels - True labels
     * @returns {SVM} Precision score
     */
    _precision(features, labels) {
        this._confusion(features, labels);
        const TP = this.confusion.TP;
        const FP = this.confusion.FP;
        if (TP + FP === 0) return this;
        this.metrics['precision'] = TP / (TP + FP);
        return this;
    }

    /**
     * Calculates Recall (Sensitivity)
     * @param {Array} features - Array of feature vectors
     * @param {Array} labels - True labels
     * @returns {Number} Recall score
     */
    _recall(features, labels) {
        this._confusion(features, labels);
        const TP = this.confusion.TP;
        const FN = this.confusion.FN;
        if (TP + FN === 0) return this;
        this.metrics['recall'] = TP / (TP + FN);
        return this;
    }

    /**
     * Calculates F1 Score (Harmonic mean of Precision and Recall)
     * @returns {SVM} F1 score
     */
    _F1Score() {
        const prec = this.metrics.precision;
        const rec = this.metrics.recall;
        if (prec + rec === 0) return this;
        this.metrics['F1'] = 2 * (prec * rec) / (prec + rec);
        return this;
    }

    /**
     * Defines the kernel function based on selected type and options
     * @returns {Function} The kernel function
     */
    _kernelFunction() {
        const type = this.options.kernel;
        const opts = this.options.kernelOptions;
        return (x, y) => {
            switch (type) {
                case 'linear':
                    return this._dot(x, y);
                case 'polynomial':
                    const degree = opts.degree || 3;
                    const coef0 = opts.coef0 || 1;
                    const gamma = opts.gamma || 1;
                    return Math.pow(gamma * this._dot(x, y) + coef0, degree);
                case 'rbf':
                    const sigma = opts.sigma || 0.5;
                    const diff = this._vectorDiff(x, y);
                    const normSq = this._vectorDot(diff, diff);
                    return Math.exp(-normSq / (2 * sigma * sigma));
                case 'sigmoid':
                    const gammaSigmoid = opts.gamma || 0.5;
                    const coef0Sigmoid = opts.coef0 || 0;
                    return Math.tanh(gammaSigmoid * this._dot(x, y) + coef0Sigmoid);
                default:
                    throw new Error(`Unknown kernel type: ${type}`);
            }
        };
    }

    /**
     * Helper for dot product of two vectors
     * @param {Array} a - First vector
     * @param {Array} b - Second vector
     * @returns {Number} Dot product
     */
    _dot(a, b) {
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /**
     * Helper to compute difference between two vectors
     * @param {Array} a - First vector
     * @param {Array} b - Second vector
     * @returns {Array} Difference vector
     */
    _vectorDiff(a, b) {
        return a.map((val, idx) => val - b[idx]);
    }

    /**
     * Helper for dot product of two vectors
     * @param {Array} a - First vector
     * @param {Array} b - Second vector
     * @returns {Number} Dot product
     */
    _vectorDot(a, b) {
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /**
     * Applies whitening normalization to features
     * @param {Array} features - Feature vector
     * @returns {Array} Whitened features
     */
    _applyWhitening(features) {
        if (!this.minMax) {
            throw new Error('Whitening parameters not initialized');
        }
        const whitened = [];
        for (let j = 0; j < features.length; j++) {
            const min = this.minMax[j].min;
            const max = this.minMax[j].max;
            whitened[j] = (features[j] - min) / (max - min);
        }
        return whitened;
    }

    /**
     * Trains the SVM model with provided features and labels
     * @param {Array} features - Array of feature vectors
     * @param {Array} labels - Original labels
     */
    train(features, labels) {
        if (features.length !== labels.length) {
            throw new Error('Features and labels length mismatch');
        }
        if (features.length < 2) {
            throw new Error('Need at least 2 samples to train');
        }

        this._trained = false;
        this._loaded = false;

        // Encode labels to +1/-1
        const encodedLabels = this._encodeLabels(labels);
        this.Y = encodedLabels;
        this.X = features.slice();

        const N = this.Y.length;
        const D = this.X[0].length;

        // Normalize data if whitening is enabled
        if (this.options.whitening) {
            this.minMax = new Array(D);
            for (let j = 0; j < D; j++) {
                const col = [];
                for (let i = 0; i < N; i++) {
                    col.push(this.X[i][j]);
                }
                const min = Math.min(...col);
                const max = Math.max(...col);
                this.minMax[j] = { min, max };
            }
            this.X = this.X.map(f => this._applyWhitening(f));
        } else {
            this.minMax = null;
        }

        // Initialize alpha coefficients
        const alphas = new Array(N).fill(0);
        this.alphas = alphas;
        this.b = 0;

        const kernel = this._kernelFunction();

        // Precompute Kernel matrix for efficiency
        const K = new Array(N);
        for (let i = 0; i < N; i++) {
            K[i] = new Array(N);
            for (let j = 0; j < N; j++) {
                K[i][j] = kernel(this.X[i], this.X[j]);
            }
        }

        // SMO Algorithm main loop
        let passes = 0;
        let iter = 0;

        while (passes < this.options.maxPasses && iter < this.options.maxIterations) {
            let numChangedAlphas = 0;
            for (let i = 0; i < N; i++) {
                const Ei = this._marginOnePrecomputed(i, K) - this.Y[i];

                // Check if sample violates KKT conditions
                if (
                    (this.Y[i] * Ei < -this.options.tol && this.alphas[i] < this.options.cost) ||
                    (this.Y[i] * Ei > this.options.tol && this.alphas[i] > 0)
                ) {
                    // Select j randomly different from i
                    let j = i;
                    while (j === i) {
                        j = Math.floor(this.options.random() * N);
                    }
                    const Ej = this._marginOnePrecomputed(j, K) - this.Y[j];

                    const alphaIold = this.alphas[i];
                    const alphaJold = this.alphas[j];

                    // Compute bounds L and H for alpha_j
                    let L, H;
                    if (this.Y[i] === this.Y[j]) {
                        L = Math.max(0, alphaJold + alphaIold - this.options.cost);
                        H = Math.min(this.options.cost, alphaJold + alphaIold);
                    } else {
                        L = Math.max(0, alphaJold - alphaIold);
                        H = Math.min(this.options.cost, this.options.cost + alphaJold - alphaIold);
                    }
                    if (Math.abs(L - H) < 1e-4) continue;

                    const eta = 2 * K[i][j] - K[i][i] - K[j][j];
                    if (eta >= 0) continue;

                    // Compute new alpha_j
                    let alphaJnew = alphaJold - (this.Y[j] * (Ei - Ej)) / eta;
                    // Clip alpha_j
                    if (alphaJnew > H) alphaJnew = H;
                    if (alphaJnew < L) alphaJnew = L;

                    if (Math.abs(alphaJnew - alphaJold) < 1e-4) continue;

                    // Compute new alpha_i
                    const alphaInew = alphaIold + this.Y[i] * this.Y[j] * (alphaJold - alphaJnew);

                    // Update alphas
                    this.alphas[i] = alphaInew;
                    this.alphas[j] = alphaJnew;

                    // Compute bias
                    const b1 =
                        this.b -
                        Ei -
                        this.Y[i] * (alphaInew - alphaIold) * K[i][i] -
                        this.Y[j] * (alphaJnew - alphaJold) * K[i][j];

                    const b2 =
                        this.b -
                        Ej -
                        this.Y[i] * (alphaInew - alphaIold) * K[i][j] -
                        this.Y[j] * (alphaJnew - alphaJold) * K[j][j];

                    if (
                        this.alphas[i] > this.options.alphaTol &&
                        this.alphas[i] < this.options.cost - this.options.alphaTol
                    ) {
                        this.b = b1;
                    } else if (
                        this.alphas[j] > this.options.alphaTol &&
                        this.alphas[j] < this.options.cost - this.options.alphaTol
                    ) {
                        this.b = b2;
                    } else {
                        this.b = (b1 + b2) / 2;
                    }

                    numChangedAlphas++;
                }
            }

            iter++;
            if (numChangedAlphas === 0) {
                passes++;
            } else {
                passes = 0;
            }
        }

        if (iter >= this.options.maxIterations) {
            throw new Error('Max iterations reached');
        }

        this.iterations = iter;

        // Compute weight vector for linear kernel
        if (this.options.kernel === 'linear') {
            this.W = new Array(D).fill(0);
            for (let i = 0; i < N; i++) {
                if (this.alphas[i] > this.options.alphaTol) {
                    for (let j = 0; j < D; j++) {
                        this.W[j] += this.alphas[i] * this.Y[i] * this.X[i][j];
                    }
                }
            }
        }

        // Keep only support vectors
        this.supportVectorIdx = [];
        const newX = [];
        const newY = [];
        const newAlphas = [];
        for (let i = 0; i < N; i++) {
            if (this.alphas[i] > this.options.alphaTol) {
                this.supportVectorIdx.push(i);
                newX.push(this.X[i]);
                newY.push(this.Y[i]);
                newAlphas.push(this.alphas[i]);
            }
        }
        this.X = newX;
        this.Y = newY;
        this.alphas = newAlphas;
        this.N = this.X.length;
        this._trained = true; // Mark as trained

        // Evaluate metrics on training data
        this._confusion(features, labels);
        this._accuracy(features, labels);
        this._precision(features, labels);
        this._recall(features, labels);
        this._F1Score();
    }

    /**
     * Predicts the class label for a single sample
     * @param {Array} sample - Feature vector
     * @returns {Number} Predicted class label (original label)
     */
    predictOne(sample) {
        const margin = this.marginOne(sample);
        const predictedLabel = margin > 0 ? 1 : -1;
        return this._decodeLabel(predictedLabel);
    }

    /**
     * Predicts class labels for multiple samples
     * @param {Array} features - Array of feature vectors
     * @returns {Array} Predicted labels (original labels)
     */
    predict(features) {
        if (!this._trained && !this._loaded) {
            throw new Error('Model not trained or loaded');
        }
        if (Array.isArray(features) && Array.isArray(features[0])) {
            return features.map(f => this.predictOne(f));
        } else {
            return this.predictOne(features);
        }
    }

    /**
     * Calculates the margin (decision function) for one sample
     * @param {Array} features - Feature vector
     * @param {Boolean} noWhitening - Skip whitening if true
     * @returns {Number} Margin value
     */
    marginOne(features, noWhitening=false) {
        if (this.options.whitening && !noWhitening) {
            features = this._applyWhitening(features);
        }
        let result = this.b;
        if (this.options.kernel === 'linear' && this.W) {
            for (let i = 0; i < this.W.length; i++) {
                result += this.W[i] * features[i];
            }
        } else {
            const kernel = this._kernelFunction();
            for (let i = 0; i < this.N; i++) {
                result += this.alphas[i] * this.Y[i] * kernel(this.X[i], features);
            }
        }
        return result;
    }

    /**
     * Computes the margin for a support vector using precomputed kernel matrix
     * @param {Number} index - Index of support vector
     * @param {Array} kernelMatrix - Precomputed kernel matrix
     * @returns {Number} Margin value
     */
    _marginOnePrecomputed(index, kernelMatrix) {
        let sum = this.b;
        for (let i = 0; i < this.N; i++) {
            sum += this.alphas[i] * this.Y[i] * kernelMatrix[index][i];
        }
        return sum;
    }

    /**
     * Retrieves indices of support vectors
     * @returns {Array} Support vector indices
     */
    supportVectors() {
        if (!this._trained && !this._loaded) {
            throw new Error('Model not trained or loaded');
        }
        return this.supportVectorIdx;
    }

    /**
     * Exports the trained model to a JSON object
     * @returns {Object} Model data
     */
    toJSON() {
        if (!this._trained && !this._loaded) {
            throw new Error('Model not trained or loaded');
        }
        const model = {
            options: this.options,
            b: this.b,
            minMax: this.minMax,
            kernel: this.options.kernel,
            kernelOptions: this.options.kernelOptions,
            W: this.W ? this.W.slice() : null,
            X: this.X.slice(),
            Y: this.Y.slice(),
            alphas: this.alphas.slice(),
            labelsMap: this.labelsMap,
            inverseLabelsMap: this.inverseLabelsMap
        };
        return model;
    }

    /**
     * Loads a model from a JSON object
     * @param {Object} model - Model data
     * @returns {SVM} Instance of SVM with loaded data
     */
    static load(model) {
        const svm = new SVM(model.options);
        svm.b = model.b;
        svm.minMax = model.minMax;
        svm.W = model.W ? model.W.slice() : null;
        svm.X = model.X.slice();
        svm.Y = model.Y.slice();
        svm.alphas = model.alphas.slice();
        svm.labelsMap = model.labelsMap;
        svm.inverseLabelsMap = model.inverseLabelsMap;
        svm._loaded = true;
        svm._trained = false;
        return svm;
    }
}
