import numpy as np
from sys import float_info
import matplotlib.pyplot as plt


generate = 0
z1 = 0


def box_muller(mean, var):
    """
    Box-Muller transform - univariate Gaussian data generator
    """
    global generate
    global z1

    generate = ~generate

    if not generate:
        return mean + z1 * var

    u, v = 0.0, 0.0
    while u < float_info.epsilon:
        u = np.random.random()
        v = np.random.random()

    z0 = np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)
    z1 = np.sqrt(-2 * np.log(u)) * np.sin(2 * np.pi * v)
    return mean + z0 * var


def polybasis_linearmodel_data_generator(n, a, w):
    x = np.linspace(-10.0, 10.0, 100)
    phi = np.zeros((n, len(x)))
    for i in range(n):
        phi[i, :] = x ** i
    y = np.dot(np.transpose(w), phi)

    # plot
    fig = plt.figure()
    # draw function
    ax1 = fig.add_subplot(111)
    ax1.plot(x, np.asarray(y).reshape(-1), color="red")

    e = np.zeros(len(x))
    for i in range(len(e)):
        e[i] = box_muller(0, a)
    y += e

    # draw data
    ax1.scatter(x, np.asarray(y).reshape(-1), s=5)
    plt.show()


def linear_model_data_generator(n, a, w, x):
    phi = np.zeros((n, 1))
    for i in range(n):
        phi[i, 0] = x ** i
    y = np.dot(np.transpose(w), phi)
    e = box_muller(0, a)
    y += e

    return y
