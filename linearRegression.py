import random

import numpy as np
from normalization import Normalizer

class LinearRegression:
    MAX_ITERATIONS = 5000
    def __init__(self, numFeatures: int, learningRate: float = 0.01, tolerance: float = 10**-4):
        self.w = np.zeros(numFeatures)
        self.b = 0
        self.learningRate = learningRate
        self.normalizer = Normalizer(numFeatures)
        self.tolerance = tolerance

    def toString(self):
        return f"w = {self.w} b = {self.b}"

    def predict(self, features: np.ndarray):
        return self.normalizer.denormalize_label(self._predict_normalized(features))

    def _predict_normalized(self, features: np.ndarray):
        return self.w.dot(features) + self.b
        # return self.w.dot(self._normalize_features(features)) + self.b

    def loss(self, featuresArray: np.ndarray, expectedVals: np.ndarray):
        l = 0
        num_rows, num_cols = featuresArray.shape
        expectedVals = self.normalizer.normalize_labels(expectedVals)
        featuresArray = self.normalizer.normalize_features(featuresArray)
        for i in range(num_rows):
            l += (self._predict_normalized(featuresArray[i, :]) - expectedVals[i]) ** 2
        return l / num_rows

    def train(self, featuresArray: np.ndarray, expectedVals: np.ndarray):
        self.normalizer.init_normalization_parameters(featuresArray, expectedVals)
        featuresArray = self.normalizer.normalize_features(featuresArray)
        expectedVals = self.normalizer.normalize_labels(expectedVals)

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
        iteration = 0
        self.normalizer.init_normalization_parameters(featuresArray, expectedVals)
        num_rows, num_cols = featuresArray.shape
        featuresArray = self.normalizer.normalize_features(featuresArray)
        expectedVals = self.normalizer.normalize_labels(expectedVals)
        while True:
            prev_w = self.w
            prev_b = self.b

            for i in np.random.permutation(num_rows):
                currFeatures = featuresArray[i, :]
                currExpected = expectedVals[i]
                currPredicted = self._predict_normalized(currFeatures)
                dw = 2 * (currPredicted - currExpected) * currFeatures
                db = 2 * (currPredicted - currExpected)
                self.w = self.w - self.learningRate * dw
                self.b = self.b - self.learningRate * db

            if (abs(np.max(self.w - prev_w)) < self.tolerance and abs(self.b - prev_b) < self.tolerance) or iteration > self.MAX_ITERATIONS:
                break

            iteration += 1
