import numpy as np


class LinearRegression:
    def __init__(self, numFeatures: int, learningRate: float = 0.01):
        self.w = np.zeros(numFeatures)
        self.b = 0
        self.learningRate = learningRate

        self._normCoefVector = np.ones(numFeatures)
        self._normOffsetVector = np.zeros(numFeatures)
        self._normCoefLabel = 1
        self._normOffsetLabel = 0

    def toString(self):
        return f"w = {self.w} b = {self.b}"

    def predict(self, features: np.ndarray):
        return self._denormalize_label(self._predict_normalized(features))

    def _predict_normalized(self, features: np.ndarray):
        return self.w.dot(self._normalize_features(features)) + self.b

    def loss(self, featuresArray: np.ndarray, expectedVals: np.ndarray):
        l = 0
        num_rows, num_cols = featuresArray.shape
        expectedVals = self._normalize_labels(expectedVals)
        for i in range(num_rows):
            l += (self._predict_normalized(featuresArray[i, :]) - expectedVals[i]) ** 2
        return l / num_rows

    def _init_normalization_parameters(self, featuresArray: np.ndarray, expectedVals: np.ndarray):
        self._normCoefLabel = 1 / (np.max(expectedVals) - np.min(expectedVals))
        self._normOffsetLabel = (-1) * np.min(expectedVals) * self._normCoefLabel

        self._normCoefVector = 1 / (np.max(featuresArray, axis=0) - np.min(featuresArray, axis=0))
        self._normOffsetVector = (-1) *  np.min(featuresArray, axis=0) * self._normCoefVector

    def _normalize_features(self, featuresArray: np.ndarray):
        return featuresArray * self._normCoefVector + self._normOffsetVector

    def _normalize_labels(self, labels):
        return labels * self._normCoefLabel + self._normOffsetLabel

    def _denormalize_label(self, label):
        return (label - self._normOffsetLabel) / self._normCoefLabel

    def train(self, featuresArray: np.ndarray, expectedVals: np.ndarray):
        self._init_normalization_parameters(featuresArray, expectedVals)
        featuresArray = self._normalize_features(featuresArray)
        expectedVals = self._normalize_labels(expectedVals)

        num_rows, num_cols = featuresArray.shape
        while True:
            dw = np.zeros(num_cols)
            db = 0
            for i in range(num_rows):
                currFeatures = featuresArray[i, :]
                currExpected = expectedVals[i]
                currPredicted = self._predict_normalized(currFeatures)

                dw += 2 * (currPredicted - currExpected) * currFeatures
                db += 2 * (currPredicted - currExpected)
            if (dw < (10 ** -10) and db < (10 ** -10)):
                break

            self.w = self.w - self.learningRate * dw / num_rows
            self.b = self.b - self.learningRate * db / num_rows

    def train_stochastic(self, featuresArray: np.ndarray, expectedVals: np.ndarray):
        self._init_normalization_parameters(featuresArray, expectedVals)
        num_rows, num_cols = featuresArray.shape
        featuresArray = self._normalize_features(featuresArray)
        expectedVals = self._normalize_labels(expectedVals)
        while True:
            prev_w = self.w
            prev_b = self.b

            for i in range(num_rows):
                currFeatures = featuresArray[i, :]
                currExpected = expectedVals[i]
                currPredicted = self._predict_normalized(currFeatures)
                dw = 2 * (currPredicted - currExpected) * currFeatures
                db = 2 * (currPredicted - currExpected)
                self.w = self.w - self.learningRate * dw
                self.b = self.b - self.learningRate * db

            if abs(np.max(self.w - prev_w)) < (10 ** -10) and abs(self.b - prev_b) < (10 ** -10):
                break
