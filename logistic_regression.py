import random_data_generator as rdg
import numpy as np
import matplotlib.pyplot as plt
import sys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient_descent(w, x, y, learning_rate):
    scores = np.dot(x, w)
    predictions = sigmoid(scores)
    output_error = predictions - y
    gradient = np.dot(x.T, output_error)
    w -= learning_rate * gradient


def newton_method(w, x, y):
    predictions = sigmoid(np.dot(x, w))
    output_error = predictions - y
    gradient = np.dot(x.T, output_error)
    D = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        D[i, i] = predictions[i] * (1 - predictions[i])
    hessian = np.dot(np.dot(x.T, D), x)
    print(np.linalg.det(hessian))
    w -= np.dot(np.linalg.inv(hessian), gradient)


def log_likelihood(x, y, w):
    scores = np.dot(x, w)
    predictions = sigmoid(scores)
    ll = - np.sum((1 - y) * np.log(1 - predictions) + y * np.log(predictions))
    return ll


def logistic_regression(x, y, learning_rate):
    w = np.zeros(x.shape[1]).reshape(x.shape[1], 1)
    ll = log_likelihood(x, y, w)
    change_ratio = 1

    while change_ratio > 0.3:
        scores = np.dot(x, w)
        predictions = sigmoid(scores)
        output_error = predictions - y
        gradient = np.dot(x.T, output_error)

        D = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            D[i, i] = predictions[i] * (1 - predictions[i])
        hessian = np.dot(np.dot(x.T, D), x)

        if np.linalg.det(hessian) == 0:
            w -= learning_rate * gradient
            print("SGD")
        else:
            w -= np.dot(np.linalg.inv(hessian), gradient)
            print("Newton's method")

        tmp_ll = log_likelihood(x, y, w)
        print("\tLog-Likelihood:", tmp_ll)
        change_ratio = np.abs(ll - tmp_ll) / ll
        ll = tmp_ll

    return w


if __name__ == "__main__":
    n = 500
    mx1 = 0
    my1 = 0
    vx1 = 2
    vy1 = 1
    mx2 = 1.5
    my2 = 1.5
    vx2 = 1
    vy2 = 1

    if len(sys.argv) < 10:
        print("Usage:", sys.argv[0], "<n>", "<mx1>", "<vx1>", "<my1>", "<vy1>", "<mx2>", "<vx2>", "<my2>", "<vy2>")
        print("Use default normal distribution: x1,y1=N(0, 1), x2,y2=N(5,1)")
    else:
        n = float(sys.argv[1])
        mx1 = float(sys.argv[2])
        vx1 = float(sys.argv[3])
        my1 = float(sys.argv[4])
        vy1 = float(sys.argv[5])
        mx2 = float(sys.argv[6])
        vx2 = float(sys.argv[7])
        my2 = float(sys.argv[8])
        vy2 = float(sys.argv[9])

    x1 = np.zeros(n)
    y1 = np.zeros(n)
    x2 = np.zeros(n)
    y2 = np.zeros(n)
    for i in range(n):
        x1[i] = rdg.box_muller(mx1, vx1)
        y1[i] = rdg.box_muller(my1, vy1)
        x2[i] = rdg.box_muller(mx2, vx2)
        y2[i] = rdg.box_muller(my2, vy2)

    # x: n * d
    x = np.hstack((x1, x2))
    x = np.vstack((x, np.hstack((y1, y2))))
    x = np.vstack((x, np.ones(2*n)))
    x = x.T

    y = np.hstack((np.zeros(n), np.ones(n))).reshape(2*n, 1)

    w = logistic_regression(x, y, learning_rate=5e-5)

    final_scores = np.dot(x, w)
    preds = np.round(sigmoid(final_scores))

    print("Accuracy:", (preds == y).sum().astype(float) / len(y) * 100, "%")

    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), alpha=0.8, s=4)
    plt.show()
    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze((preds == y)), alpha=0.8, s=4)
    plt.show()
