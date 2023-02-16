#################################
# Your name: Dori Rimon
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""


def helper():
	mnist = fetch_openml('mnist_784', as_frame=False)
	data = mnist['data']
	labels = mnist['target']

	neg, pos = "0", "8"
	train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
	test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

	train_data_unscaled = data[train_idx[:6000], :].astype(float)
	train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

	validation_data_unscaled = data[train_idx[6000:], :].astype(float)
	validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

	test_data_unscaled = data[60000 + test_idx, :].astype(float)
	test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

	# Preprocessing
	train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
	validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
	test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
	return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def perceptron(data, labels):
	"""
	returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
	"""
	d = data.shape[1]
	w = np.array([0 for _ in range(d)])
	for i in range(data.shape[0]):
		x, label = data[i], int(labels[i])
		y = int(np.sign(np.dot(w, x)))
		if y == 0:
			y = 1
		if y != label:
			w = np.add(w, label * x)
	return w

def accuracy(w, data, labels):
	n = data.shape[0]
	acc = 0
	for i in range(n):
		x, label = data[i], int(labels[i])
		y = int(np.sign(np.dot(w, x)))
		if y == 0:
			y = 1
		if y == label:
			acc += 1
	return acc / n

def a():
	train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
	train_data = train_data / np.linalg.norm(train_data)
	T = 100
	nl = [5, 10, 50, 100, 500, 1000, 5000]
	acc = []
	for n in nl:
		print('running on n =', n)
		measurements = []
		for t in range(T):
			data, labels = train_data[:n], train_labels[:n]
			perm = np.random.permutation(n)
			data, labels = data[perm], labels[perm]
			w = perceptron(data, labels)
			measurements.append(accuracy(w, test_data, test_labels))
		mean = np.mean(measurements)
		p5 = np.percentile(measurements, 5)
		p95 = np.percentile(measurements, 95)
		acc.append([mean, p5, p95])
	print(np.array(acc))

def b():
	train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
	train_data = train_data / np.linalg.norm(train_data)
	w = perceptron(train_data, train_labels)
	plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
	plt.colorbar()
	plt.show()

def c():
	train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
	train_data = train_data / np.linalg.norm(train_data)
	test_data = test_data / np.linalg.norm(test_data)
	w = perceptron(train_data, train_labels)
	print(accuracy(w, test_data, test_labels))

def d():
	train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
	train_data = train_data / np.linalg.norm(train_data)
	test_data = test_data / np.linalg.norm(test_data)
	w = perceptron(train_data, train_labels)
	for i in range(test_data.shape[0]):
		x, label = test_data[i], int(test_labels[i])
		y = int(np.sign(np.dot(w, x)))
		if y != label:
			plt.imshow(np.reshape(x, (28, 28)), interpolation='nearest')
			plt.colorbar()
			plt.show()
			break


if __name__ == "__main__":
	# a()
	# b()
	# c()
	# d()
	pass
