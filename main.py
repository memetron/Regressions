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

def gen_test_linear(dimension: int, num_cases: int, noiseAmt: float):
    w = np.array([random.random() * 10 for i in range(dimension)])
    b = random.random() * 100 - 50
    features = np.empty(shape=(num_cases, dimension))
    labels = np.zeros(num_cases)
    for i in range(num_cases):
        features[i] = [random.random() * 10 - 5 for i in range(dimension)]
        labels[i] = features[i].dot(w) + b

    return (features, noise(labels, noiseAmt))

def gen_test_poly(dimension: int, degree: int, num_cases: int, noiseAmt: float):
    w = np.array([random.random() * 10 for i in range(dimension)])
    b = random.random() * 100 - 50
    features = np.empty(shape=(num_cases, dimension))
    labels = np.zeros(num_cases)
    for i in range(num_cases):
        features[i] = [random.random() * 10 - 5 for i in range(dimension)]
        labels[i] = (features[i] ** degree).dot(w) + b
    return (features, noise(labels, noiseAmt))

def split_data(features: np.ndarray, labels: np.ndarray, training_ratio: float):
    num_rows, num_cols = features.shape
    num_training: int = int(training_ratio * num_rows)

    return (features[:num_training, :], labels[:num_training]), \
           (features[num_training:, :], labels[num_training:])

def test_linear(dimension: int, numData: int, noiseAmt:float = 0.1):
    lr1 = LinearRegression(dimension, 0.01)

    (features, labels) = gen_test_poly(dimension, 2, numData, noiseAmt)
    (training, test) = split_data(features, labels, 0.9)
    (trainingFeatures, trainingLabels) = training
    (testFeatures, testLabels) = test

    lr1.train_stochastic(trainingFeatures, trainingLabels)
    print(f"model = ({lr1.toString()}), "
          f"e_training={lr1.loss(trainingFeatures, trainingLabels)}, "
          f"e_test={lr1.loss(testFeatures, testLabels)}")

    if dimension == 1: # plot if 2d graph applicable
        plt.scatter(trainingFeatures, trainingLabels, color='blue', label='Training Data')
        plt.scatter(testFeatures, testLabels, color='green', label='Test Data')
        x_values = np.linspace(min(trainingFeatures), max(trainingFeatures), 100).reshape(-1, 1)
        y_values = lr1.w * x_values + lr1.b
        plt.plot(x_values, y_values, label=f"y={lr1.w}x+{lr1.b}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression')
        plt.legend()
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
    # print(f"Starting linear test . . .")
    # start = time.time()
    # test_linear(1, 1000)
    # print(f"Linear test finished in {time.time() - start} seconds.")
    #

    print(f"Starting polynomial test . . .")
    start = time.time()
    test_poly(5, 2, 100)
    print(f"polynomial test finished in {time.time() - start} seconds.")
