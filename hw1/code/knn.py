from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

"""
DESC:   This is the code relevant to the programming assignment
"""


# Code for debug purposes
DEBUG = False
dprint = lambda x: print(x) if DEBUG else None


def knn(train, train_labels, query_image, k):
	"""
	The main knn algorithm

	:param train: train images
	:param train_labels: corresponding train labels
	:param query_image: the image we want to classify
	:param k: int, the k parameter
	:return: int, the label

	"""
	n = len(train)
	distances = []
	for i, image in enumerate(train):
		distances.append(np.linalg.norm(query_image - image))

	idx = np.argpartition(distances, k)
	idx = idx[:k]
	labels = np.array(train_labels[idx], dtype='int64')
	counts = np.bincount(labels)
	return np.argmax(counts)


def predict(train, train_labels, test, test_labels, k):
	"""
	Runs the KNN algorithm on multiple test images

	:param train: train images
	:param train_labels: corresponding train labels
	:param test: test images
	:param test_labels: corresponding test labels
	:param k: int, the k parameter
	:return: 0 <= float <= 1, the accuracy of the predictions

	"""
	n = len(test)
	res = 0
	for i, image in enumerate(test):
		prediction = knn(train, train_labels, image, k)
		dprint('prediction: ' + str(prediction) + '; actual: ' + str(test_labels[i]))
		if int(prediction) == int(test_labels[i]):
			res += 1

	return res / n


def plot_k_accuracy(train, train_labels, test, test_labels, max_k):
	"""
	Test accuracy on increasing values of k - creates a plot

	:param train: train images
	:param train_labels: corresponding train labels
	:param test: test images
	:param test_labels: corresponding test labels
	:param max_k: int, the maximum k to test up to

	"""
	x = [i + 1 for i in range(max_k)]
	y = []
	for i in range(max_k):
		res = predict(train, train_labels, test, test_labels, i + 1)
		y.append(res)

	plt.scatter(x, y)
	plt.title('Prediction accuracy as a function of k')
	plt.xlabel('k')
	plt.ylabel('Accuracy')
	plt.show()


def plot_n_accuracy(train, train_labels, test, test_labels, max_n):
	"""
	Test accuracy on increasing values of n - creates a plot
	
	:param train: train images
	:param train_labels: corresponding train labels
	:param test: test images
	:param test_labels: corresponding test images
	:param max_n: the maximum n to test up to

	"""
	x = [n for n in range(100, max_n + 1, 100)]
	y = []
	for n in range(100, max_n + 1, 100):
		res = predict(train[:n], train_labels[:n], test, test_labels, 1)
		y.append(res)

	plt.scatter(x, y)
	plt.title('Prediction accuracy as a function of n')
	plt.xlabel('n')
	plt.ylabel('Accuracy')
	plt.show()


def main():
	"""
	The main function - runs the code of the programming assignment
	
	"""
	mnist = fetch_openml('mnist_784')
	data = mnist['data']
	labels = mnist['target']

	idx = np.random.RandomState(0).choice(70_000, 11_000)
	train = data[idx[:10_000], :].astype(int)
	train_labels = labels[idx[:10_000]]
	test = data[idx[10_000:], :].astype(int)
	test_labels = labels[idx[10_000:]]

	print(predict(train[:1_000], train_labels[:1_000], test, test_labels, 10))
	plot_k_accuracy(train[:1_000], train_labels[:1_000], test, test_labels, 100)
	plot_n_accuracy(train, train_labels, test, test_labels, 5_000)


if __name__ == "__main__":
	main()
