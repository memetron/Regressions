# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from linearRegression import LinearRegression
from polynomialRegression import PolynomialRegression

def noise(features: np.ndarray, noiseAmt: float):
    noisyFeatures = np.zeros(len(features))
    for i in range(len(features)):
        noisyFeatures[i] = features[i] * ((1 - noiseAmt) + random.random() * (2 * noiseAmt))
    return noisyFeatures

# generate a linear curve with some noise
def gen_test_linear(dimension: int, num_cases: int, noiseAmt: float):
    w = np.array([random.random() * 10 for i in range(dimension)])
    b = random.random() * 100 - 50
    features = np.empty(shape=(num_cases, dimension))
    labels = np.zeros(num_cases)
    for i in range(num_cases):
        features[i] = [random.random() * 10 - 5 for i in range(dimension)]
        labels[i] = features[i].dot(w) + b

    return (features, noise(labels, noiseAmt))

# generate a polynomial curve with some added noise
def gen_test_poly(dimension: int, degree: int, num_cases: int, noiseAmt: float):
    w = np.array([random.random() * 10 for i in range(dimension)])
    b = random.random() * 100 - 50
    features = np.empty(shape=(num_cases, dimension))
    labels = np.zeros(num_cases)
    for i in range(num_cases):
        features[i] = [random.random() * 10 - 5 for i in range(dimension)]
        labels[i] = (features[i] ** degree).dot(w) + b
    return (features, noise(labels, noiseAmt))

# splits a set of data into a training and test set
def split_data(features: np.ndarray, labels: np.ndarray, training_ratio: float):
    num_rows, num_cols = features.shape
    num_training: int = int(training_ratio * num_rows)

    return (features[:num_training, :], labels[:num_training]), \
           (features[num_training:, :], labels[num_training:])

def test_linear(dimension: int, numData: int, noiseAmt:float = 0.1):
    (features, labels) = gen_test_linear(dimension, numData, noiseAmt)
    (training, test) = split_data(features, labels, 0.9)
    (trainingFeatures, trainingLabels) = training
    (testFeatures, testLabels) = test

    start = time.time()
    model = LinearRegression(dimension)
    model.train_batch(trainingFeatures, trainingLabels, 32)
    print(f"Actual ({trainingFeatures[0]}, {trainingLabels[0]}), Predicted : ({trainingFeatures[0]}, {model.predict(trainingFeatures[0])})")
    print(f"batch 100 trained in {time.time() - start} seconds")
    print(f"model = ({model.toString()}), "
          f"e_training={model.loss(trainingFeatures, trainingLabels)}, "
          f"e_test={model.loss(testFeatures, testLabels)}")

    plt.scatter(testFeatures, testLabels)
    x_pred, y_pred = model.get_line(np.min(trainingFeatures), np.max(trainingFeatures))
    plt.plot(x_pred, y_pred)
    plt.show()


def test_poly(dimension: int, degree: int, numData: int, noiseAmt:float = 0.1):
    model = PolynomialRegression(dimension, degree)
    (features, labels) = gen_test_poly(dimension, degree, numData, noiseAmt)
    (training, test) = split_data(features, labels, 0.9)
    (trainingFeatures, trainingLabels) = training
    (testFeatures, testLabels) = test

    model.train(trainingFeatures, trainingLabels)
    print(f"model = ({model.toString()}), "
          f"e_training={model.loss(trainingFeatures, trainingLabels)}, "
          f"e_test={model.loss(testFeatures, testLabels)}")


if __name__ == '__main__':
    print(f"Starting linear test . . .")
    start = time.time()
    test_linear(1, 500)
    print(f"Linear test finished in {time.time() - start} seconds.")


    print(f"Starting polynomial test . . .")
    start = time.time()
    test_poly(5, 2, 10000)
    print(f"polynomial test finished in {time.time() - start} seconds.")
