import random

import numpy as np
from normalization import Normalizer

class LinearRegression:
    _MAX_ITERATIONS: int = 5000
    _TOLERANCE: float = 0.001
    _LEARNING_RATE: float = 0.01


    def __init__(self, numFeatures: int):
        self.w = np.zeros(numFeatures)
        self.b = 0
        self.normalizer = Normalizer(numFeatures)


    def toString(self):
        return f"w = {self.w} b = {self.b}"

    # denormalized prediction
    # returns a denormalized label given a denormalized features vector
    def predict(self, features: np.ndarray):
        return self.normalizer.denormalize_label(self._predict_normalized(features))

    # Gives a normalized prediction when fed a normalized features vector
    # Much faster than the denormalized predict, since no normalization occurs here
    def _predict_normalized(self, features: np.ndarray):
        return self.w.dot(features) + self.b

    # returns a normalized loss over a given test set
    def loss(self, featuresArray: np.ndarray, expectedVals: np.ndarray):
        l = 0
        num_rows, num_cols = featuresArray.shape
        expectedVals = self.normalizer.normalize_labels(expectedVals)
        featuresArray = self.normalizer.normalize_features(featuresArray)
        for i in range(num_rows):
            l += (self._predict_normalized(featuresArray[i, :]) - expectedVals[i]) ** 2
        return l / num_rows

    # Train model using non-stochastic gradient descent
    # Normalizer is configured based off the initial training set, and data is normalized before training
    def train(self, featuresArray: np.ndarray, expectedVals: np.ndarray):
        # calibrate normalizer based off of training set, then normalize data
        self.normalizer.init_normalization_parameters(featuresArray, expectedVals)
        featuresArray = self.normalizer.normalize_features(featuresArray)
        expectedVals = self.normalizer.normalize_labels(expectedVals)

        num_rows, num_cols = featuresArray.shape
        # iterate until convergence
        while True:
            dw = np.zeros(num_cols)
            db = 0
            # calc gradient over all data
            for i in range(num_rows):
                currFeatures = featuresArray[i, :]
                currExpected = expectedVals[i]
                currPredicted = self._predict_normalized(currFeatures)

                dw += 2 * (currPredicted - currExpected) * currFeatures
                db += 2 * (currPredicted - currExpected)
            if (dw < (10 ** -10) and db < (10 ** -10)):
                break

            self.w = self.w - self._LEARNING_RATE * dw / num_rows
            self.b = self.b - self._LEARNING_RATE * db / num_rows

    # trains the regression using stochastic gradient descent
    def train_stochastic(self, featuresArray: np.ndarray, expectedVals: np.ndarray):
        # calibrate normalizer based off of training set, then normalize data
        self.normalizer.init_normalization_parameters(featuresArray, expectedVals)
        featuresArray = self.normalizer.normalize_features(featuresArray)
        expectedVals = self.normalizer.normalize_labels(expectedVals)

        num_rows, num_cols = featuresArray.shape
        iteration = 0
        # iterate until convergence
        while True:
            # change in model parameters will be tracked over the epoch to check for convergence
            prev_w = self.w
            prev_b = self.b

            # shuffle rows each epoch to reduce bias
            for i in np.random.permutation(num_rows):
                # calc gradient relative to a single datum
                currFeatures = featuresArray[i, :]
                currExpected = expectedVals[i]
                currPredicted = self._predict_normalized(currFeatures)
                dw = 2 * (currPredicted - currExpected) * currFeatures
                db = 2 * (currPredicted - currExpected)
                self.w = self.w - self._LEARNING_RATE * dw
                self.b = self.b - self._LEARNING_RATE * db

            # test if convergence criteria met
            if (abs(np.max(self.w - prev_w)) < self._TOLERANCE and abs(self.b - prev_b) < self._TOLERANCE) or iteration > self._MAX_ITERATIONS:
                break

            iteration += 1
