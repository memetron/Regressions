import math
import random

import numpy as np
from normalization import Normalizer

class LinearRegression:
    _MAX_ITERATIONS: int = 50000
    _TOLERANCE: float = 10 ** (-5)
    _LEARNING_RATE: float = 0.01
    _DECAY_RATE: float = 0.001

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
        return np.dot(features, self.w) + self.b

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
        iteration = 0
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

            self.w = self.w - self._LEARNING_RATE * dw / num_rows
            self.b = self.b - self._LEARNING_RATE * db / num_rows
            if (abs(np.max(dw)) < self._TOLERANCE and abs(db) < self._TOLERANCE) or iteration > self._MAX_ITERATIONS:
                break
            iteration += 1

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
            adaptiveLearningRate = self._LEARNING_RATE / (1 + self._DECAY_RATE * iteration)
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
                self.w = self.w - adaptiveLearningRate * dw
                self.b = self.b - adaptiveLearningRate * db

            # test if convergence criteria met
            if (abs(np.max(self.w - prev_w)) < self._TOLERANCE and abs(self.b - prev_b) < self._TOLERANCE) or iteration > self._MAX_ITERATIONS:
                break

            iteration += 1

    def train_batch(self, featuresArray: np.ndarray, expectedVals: np.ndarray, batch_size: int):
        # calibrate normalizer based off of training set, then normalize data
        self.normalizer.init_normalization_parameters(featuresArray, expectedVals)
        featuresArray = self.normalizer.normalize_features(featuresArray)
        expectedVals = self.normalizer.normalize_labels(expectedVals)

        num_rows, num_cols = featuresArray.shape
        iteration = 0
        # iterate until convergence
        while True:
            # Adaptive learning rate calculation
            adaptiveLearningRate = self._LEARNING_RATE / (1 + self._DECAY_RATE * iteration)

            # change in model parameters will be tracked over the epoch to check for convergence
            prev_w = self.w.copy()
            prev_b = self.b

            # shuffle data each epoch to reduce bias
            indices = np.random.permutation(num_rows)
            gradient_norm = 0
            for i in range(0, num_rows, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_features = featuresArray[batch_indices, :]
                batch_expected = expectedVals[batch_indices]

                # predict for the batch
                batch_predicted = self._predict_normalized(batch_features)
                # compute gradients for the batch
                dw = 2 * np.dot((batch_predicted - batch_expected).T, batch_features) / batch_size
                db = 2 * np.sum(batch_predicted - batch_expected) / batch_size

                # update parameters using adaptive learning rate
                self.w = self.w - adaptiveLearningRate * dw
                self.b = self.b - adaptiveLearningRate * db

            # test if convergence criteria met
            if (np.max(np.abs(self.w - prev_w)) < self._TOLERANCE and abs(
                    self.b - prev_b) < self._TOLERANCE) or iteration > self._MAX_ITERATIONS:
                break

            iteration += 1
