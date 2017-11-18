from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import os
import random_data_generator as rdg


def show_digits(img, label):
    """
    Show 0~9 digit in one image.
    :param img: mnist image
    :param label: mnist label
    :return:
    """
    l = int(np.sqrt(img.shape[1]))
    plt.figure(num='mnist', figsize=(14, 6))
    for i in range(10):
        j = 0
        while label[j] != i:
            j = j + 1
        curr_img = img[j].reshape((l, l))
        curr_label = label[j]
        plt.subplot(2, 5, i + 1)
        plt.title("Cluster is " + str(curr_label))
        plt.imshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.show()


def seperate_data_by_label(imgs, labels):
    seperated = {}
    for i in range(len(labels)):
        if labels[i] not in seperated:
            seperated[labels[i]] = []
        seperated[labels[i]].append(imgs[i])
    return seperated


def im2col_indices(img, stride):
    pool_size = stride * stride
    x_col = np.zeros((pool_size, int(img.shape[0] * img.shape[1] / pool_size)))
    k = 0
    for i in range(0, img.shape[0], stride):
        for j in range(0, img.shape[1], stride):
            x_col[0, k] = img[i, j]
            x_col[1, k] = img[i, j+1]
            x_col[2, k] = img[i+1, j]
            x_col[3, k] = img[i+1, j+1]

            k += 1
    return x_col


def max_pooling(img, stride):
    x_col = im2col_indices(img, stride)
    max_idx = np.argmax(x_col, axis=0)
    out = x_col[max_idx, range(max_idx.size)]
    out = out.reshape(int(img.shape[0]/2), int(img.shape[1]/2))
    return out


def bernoulli_likelihood(mu, x):
    l = 1
    for i in range(len(x)):
        l *= mu[i] ** x[i] * (1 - mu[i]) ** (1 - x[i])
    return l


def naive_bayes(pi, mu, x):
    predictions = np.zeros(len(x))
    for i in range(len(x)):
        print("Testing %dth image." % i)
        likelihood = np.zeros(10)
        for k in range(num_label):
            likelihood[k] = pi[k] * bernoulli_likelihood(mu[:, k], x[i])
        predictions[i] = np.argmax(likelihood)
        # print(likelihood)
    return predictions


def clustering(predictions, labels):
    clusters = np.zeros((10, 10), dtype=np.int)
    for index, item in enumerate(predictions):
        clusters[labels[index], int(item)] += 1
    return clusters


if __name__ == "__main__":

    cwd = os.getcwd()
    mndata = MNIST(cwd + '/data')
    # train_img, train_label = mndata.load_training()
    test_img, test_label = mndata.load_testing()

    # train_img_ndarray = np.array(train_img)
    # train_label_ndarray = np.array(train_label)
    test_img_ndarray = np.array(test_img)
    test_label_ndarray = np.array(test_label)
    # show_digits(test_img_ndarray, test_label_ndarray)

    num_label = 10

    print("Building input x...")
    tmp_x = np.zeros((len(test_img_ndarray), 28, 28))
    for i in range(len(tmp_x)):
        tmp_x[i] = test_img_ndarray[i].reshape(28, 28)
    x = np.zeros((len(test_img_ndarray), 196))
    for i in range(len(x)):
        x[i] = max_pooling(tmp_x[i], 2).reshape(196)

    x = np.round(x / 256)
    # show_digits(x, test_label_ndarray)

    num_img = x.shape[0]
    num_pixel = x.shape[1]

    pi = np.zeros(num_label)
    for i in range(num_label):
        while pi[i] <= 0 or pi[i] >= 1:
            pi[i] = rdg.box_muller(0.5, 0.1)

    mu = np.zeros((num_pixel, num_label))
    for i in range(num_pixel):
        for j in range(num_label):
            while mu[i, j] <= 0 or mu[i, j] >= 1:
                mu[i, j] = rdg.box_muller(0.5, 0.1)

    likelihood = np.zeros((num_label, num_img))
    w = np.zeros_like(likelihood)

    distance = 1000

    print("Running EM Algorithm...")
    while distance > 2:
        old_mu = np.copy(mu)
        old_pi = np.copy(pi)
        for k in range(num_label):
            for n in range(num_img):
                likelihood[k, n] = pi[k] * bernoulli_likelihood(mu[:, k], x[n])

        for n in range(num_img):
            w[:, n] = likelihood[:, n] / sum(likelihood[:, n])

        tmp_mu = np.zeros_like(mu)
        for k in range(num_label):
            nm = sum(w[k])
            for n in range(num_img):
                tmp_mu[:, k] += w[k, n] * x[n]
            tmp_mu[:, k] /= nm
            pi[k] = nm / num_img
        mu = np.copy(tmp_mu)

        distance = np.linalg.norm(mu - old_mu) + np.linalg.norm(pi - old_pi)
        print("theta changes:", distance)

    print("Clustering...")
    # print("Predicting...")
    predictions = naive_bayes(pi, mu, x)
    # print(predictions)
    # print(test_label_ndarray[0:100])
    clusters = clustering(predictions, test_label_ndarray)
    clusters_label = np.zeros(10)
    for i in range(10):
        print(i, ":", clusters[i])
        clusters_label[i] = np.argmax(clusters[i])
    print(clusters_label)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for k in range(num_label):
        for index, item in enumerate(predictions):
            if clusters_label[int(item)] == k:
                if test_label_ndarray[index] == k:
                    TP += 1
                else:
                    FP += 1
            else:
                if test_label_ndarray[index] == k:
                    FN += 1
                else:
                    TN += 1

    print("confusion matrix:")
    print("\tTrue Positive:", TP)
    print("\tTrue Negative:", TN)
    print("\tFalse Positive:", FP)
    print("\tFalse Negative:", FN)
    print("Sensitivity:", TP / (TP + FN))
    print("Specificity:", TN / (TN + FP))

    bin_mu = np.zeros_like(mu)
    for n in range(num_pixel):
        for k in range(num_label):
            if mu[n, k] <= 0.2:
                bin_mu[n, k] = 0
            elif mu[n, k] <= 0.5:
                bin_mu[n, k] = 128
            elif mu[n, k] <= 0.7:
                bin_mu[n, k] = 200
            else:
                bin_mu[n, k] = 255

    for k in range(num_label):
        cluster_label = np.arange(10)
        show_digits(bin_mu.T, cluster_label)