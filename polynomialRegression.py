from itertools import combinations_with_replacement

import numpy as np
import math
from linearRegression import LinearRegression


class PolynomialRegression:
    def __init__(self, numFeatures: int, degree: int, learningRate: float = 0.01):
        self.numFeatures = numFeatures
        self.degree = degree
        numPolyFeatures = math.comb(numFeatures + degree, numFeatures) - 1  # find number of unique polynomial terms
        self._model = LinearRegression(numPolyFeatures, learningRate)

        self._normCoefVector = np.ones(numFeatures)
        self._normOffsetVector = np.zeros(numFeatures)
        self._normCoefLabel = 1
        self._normOffsetLabel = 0

    def train(self, featuresArray: np.ndarray, expectedVals: np.ndarray):
        self._init_normalization_parameters(featuresArray, expectedVals)
        featuresArray = self._normalize_features(featuresArray)
        expectedVals = self._normalize_labels(expectedVals)
        self._model.train_stochastic(self._transform_mult(featuresArray), expectedVals)

    def predict(self, features: np.ndarray):
        features = self._normalize_features(features)
        return self._denormalize_label(self._model._predict_normalized(self._transform(features)))
        # return self._model._predict_normalized(self._transform(features))

    def loss(self, featuresArray: np.ndarray, expectedVals: np.ndarray):
        featuresArray = self._normalize_features(featuresArray)
        expectedVals = self._normalize_labels(expectedVals)
        return self._model.loss(self._transform_mult(featuresArray), expectedVals)

    def toString(self):
        return f"degree={self.degree}, {self._model.toString()}"

    def _transform_mult(self, features: np.ndarray):
        n_samples, n_features = features.shape
        transformed_features = []
        for i in range(1, self.degree + 1):
            for indices in combinations_with_replacement(range(n_features), i):
                new_feature = np.prod(features[:, indices], axis=1, keepdims=True)
                transformed_features.append(new_feature)
        return np.hstack(transformed_features)

    def _transform(self, features: np.ndarray):
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return np.hstack(self._transform_mult(features)).flatten()  # Flatten since we're dealing with a single sample

    def _init_normalization_parameters(self, featuresArray: np.ndarray, expectedVals: np.ndarray):
        self._normCoefLabel = 1 / (np.max(expectedVals) - np.min(expectedVals))
        self._normOffsetLabel = (-1) * np.min(expectedVals) * self._normCoefLabel

        self._normCoefVector = 1 / (np.max(featuresArray, axis=0) - np.min(featuresArray, axis=0))
        self._normOffsetVector = (-1) * np.min(featuresArray, axis=0) * self._normCoefVector

    def _normalize_features(self, featuresArray: np.ndarray):
        return featuresArray * self._normCoefVector + self._normOffsetVector

    def _normalize_labels(self, labels):
        return labels * self._normCoefLabel + self._normOffsetLabel

    def _denormalize_label(self, label):
        return (label - self._normOffsetLabel) / self._normCoefLabel
