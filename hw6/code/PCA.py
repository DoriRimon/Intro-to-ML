import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np

def plot_vector_as_image(image, h, w):
	"""
	utility function to plot a vector as image.
	Args:
	image - vector of pixels
	h, w - dimesnions of original pi
	"""
	plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
	plt.title('title', size=12)
	plt.show()

def get_pictures_by_name(name='Ariel Sharon'):
	"""
	Given a name returns all the pictures of the person with this specific name.
	YOU CAN CHANGE THIS FUNCTION!
	THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
	"""
	lfw_people = load_data()
	selected_images = []
	n_samples, h, w = lfw_people.images.shape
	target_label = list(lfw_people.target_names).index(name)
	for image, target in zip(lfw_people.images, lfw_people.target):
		if target == target_label:
			image_vector = image.reshape((h*w, 1))
			selected_images.append(image_vector)
	return selected_images, h, w

def load_data():
	# Don't change the resize factor!!!
	lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
	return lfw_people


"""
Other then the PCA function below the rest of the functions are yours to change.
"""

def PCA(X, k):
	"""
	Compute PCA on the given matrix.

	Args:
		X - Matrix of dimensions (n,d). Where n is the number of sample points and d is the dimension of each sample.
		For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimension of the matrix would be (10,100).
		k - number of eigenvectors to return

	Returns:
	  U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors
	  		of the covariance matrix.
	  S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
	"""
	sigma = len(X) * (np.dot(np.transpose(X), X))
	u, s, vh = np.linalg.svd(sigma)

	S = s[:k]
	U = vh[:k]

	return U, S

def main():
	images, h, w = get_pictures_by_name()
	X = np.reshape(images, (np.shape(images)[0], np.shape(images)[1]))
	U, S = PCA(X, 10)

	# sub_question 2
	f, ax_arr = plt.subplots(2, 5)
	for i in range(len(U)):
		y = i % 5
		x = 0 if i < 5 else 1
		v = U[i]
		ax_arr[x, y].imshow(v.reshape(h, w), cmap=plt.cm.gray)
		ax_arr[x, y].axis('off')

	# plt.show()

	# sub-question 3
	k_vals = [1, 5, 10, 30, 50, 100]
	distances = []

	for k in k_vals:
		U, S = PCA(X, k)
		f, ax_arr = plt.subplots(2, 5)
		n = len(X)
		rand_ind = np.random.randint(n, size=5)
		rand_img = X[rand_ind, :]

		U = np.transpose(U)

		current = 0

		for i in range(5):
			x = rand_img[i]
			xhat = (U @ np.transpose(U)) @ rand_img[i, :]

			ax_arr[0, i].imshow(x.reshape(h, w), cmap=plt.cm.gray)
			ax_arr[1, i].imshow(xhat.reshape(h, w), cmap=plt.cm.gray)
			ax_arr[0, i].axis('off')
			ax_arr[1, i].axis('off')

			current += np.linalg.norm(xhat - x)

		distances.append(current)

	# plt.show()

	# Code to plot the k - l2 graph
	plt.plot(k_vals, distances, label='The l2 distance as a function of k')
	plt.xlabel('k')
	plt.ylabel('l2')

	# plt.show()


if __name__ == "__main__":
	main()
