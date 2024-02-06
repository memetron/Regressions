from itertools import combinations_with_replacement

import numpy as np
import math
from linearRegression import LinearRegression

# wraps a linear regression with a polynomial trainsform of given degree
# e.g. if your feature vector is originally <x,y> it would become <x,y,x^2,y^2,xy>
# and the linear model would be trained off the transformed features
class PolynomialRegression:
    def __init__(self, numFeatures: int, degree: int):
        self.numFeatures = numFeatures
        self.degree = degree
        numPolyFeatures = math.comb(numFeatures + degree, numFeatures) - 1  # find number of unique polynomial terms
        self._model = LinearRegression(numPolyFeatures)

    # Wrappers for linear regression functions using a polynomial transform
    def train(self, featuresArray: np.ndarray, expectedVals: np.ndarray):
        self._model.train_batch(self._transform_mult(featuresArray), expectedVals, 128)

    def predict(self, features: np.ndarray):
        return self._model.predict(self._transform_mult(features))

    def loss(self, featuresArray: np.ndarray, expectedVals: np.ndarray):
        return self._model.loss(self._transform_mult(featuresArray), expectedVals)

    def toString(self):
        return f"degree={self.degree}, {self._model.toString()}"

    # performs a polynomial transform on a set of feature vectors
    def _transform_mult(self, features: np.ndarray):
        n_samples, n_features = features.shape
        transformed_features = []
        for i in range(1, self.degree + 1):
            for indices in combinations_with_replacement(range(n_features), i):
                new_feature = np.prod(features[:, indices], axis=1, keepdims=True)
                transformed_features.append(new_feature)
        return np.hstack(transformed_features)

    def get_line(self, l, r):
        x = np.linspace(l, r, 100).reshape(-1, 1)
        return x, self.predict(x)