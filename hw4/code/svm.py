#################################
# Your name: Dori Rimon
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]

def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """

    linear = svm.SVC(C=1000, kernel='linear')
    quadratic = svm.SVC(C=1000, kernel='poly', degree=2)
    rbf = svm.SVC(C=1000, kernel='rbf')

    svm_list = [linear, quadratic, rbf]
    for s in svm_list:
        s.fit(X_train, y_train)
        # Code to create the graphs:
        # create_plot(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)), s)
        # plt.show()

    return np.array([s.n_support_ for s in svm_list])

def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """

    C_list = [10 ** i for i in range(-5, 6)]

    train_accuracy = []
    validation_accuracy = []

    optimum_accuracy = 0

    for C in C_list:
        s = svm.SVC(C=C, kernel='linear')
        s.fit(X_train, y_train)

        accuracy = calculate_accuracy(X_train, y_train, s)
        train_accuracy.append(accuracy)

        validation = calculate_accuracy(X_val, y_val, s)
        validation_accuracy.append(validation)

        if validation > optimum_accuracy:
            optimum_accuracy = validation

    # Code to plot the graph:
    # plt.plot([i for i in range(-5, 6)], train_accuracy, c='g', label='accuracy on the training set')
    # plt.plot([i for i in range(-5, 6)], validation_accuracy, c='r', label='accuracy on the validation set')
    # plt.title("Linear SVM accuracy as a function of C")
    # plt.xlabel("log_10 of C (C = 10 ^ i)")
    # plt.ylabel("accuracy")
    # plt.legend(loc='best')

    return np.array(validation_accuracy)

def calculate_accuracy(X, y, svm_model):
    res = 0

    for i in range(len(X)):
        h_x = svm_model.predict([X[i]])[0]
        if h_x == y[i]:
            res += 1

    return res / len(X)

def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    # TODO: add your code here

def main():
    X_train, y_train, X_val, y_val = get_points()
    linear_accuracy_per_C(X_train, y_train, X_val, y_val)
    train_three_kernels(X_train, y_train, X_val, y_val)

# if __name__ == '__main__':
#     main()

