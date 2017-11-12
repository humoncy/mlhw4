from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def show_digits(img, label):
    """
    Show 0~9 digit in one image.
    :param img: mnist image
    :param label: mnist label
    :return:
    """
    plt.figure(num='mnist', figsize=(14, 6))
    for i in range(10):
        j = 0
        while train_label[j] != i:
            j = j + 1
        curr_img = np.reshape(train_img[j], (28, 28))
        curr_label = train_label[j]
        plt.subplot(2, 5, i + 1)
        plt.title("Label is " + str(curr_label))
        plt.imshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.show()


if __name__ == "__main__":

    cwd = os.getcwd()
    mndata = MNIST(cwd + '/data')
    train_img, train_label = mndata.load_training()
    test_img, test_label = mndata.load_testing()
    # show_digits(train_img, train_label)