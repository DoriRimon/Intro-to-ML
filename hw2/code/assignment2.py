#################################
# Your name: Dori Rimon
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals

# TODO - remove imports
DEBUG = True
dprint = lambda exp: print(exp) if DEBUG else None


class Assignment2(object):
	"""Assignment 2 skeleton.

	Please use these function signatures for this assignment and submit this file, together with the intervals.py.
	"""

	def __init__(self):
		def p_func(y, x):
			"""
			Computes P(Y = y | X = x)

			:param y:
			:param x:
			:return:
			"""
			if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
				return 0.8 if y == 1 else 0.2
			else:
				return 0.1 if y == 1 else 0.9

		def zo(hx, y):
			return 0 if hx == y else 1

		self.p = p_func
		self.zo = zo

		def srm_penalty(VCdim, n, delta=0.1):
			return 2 * np.sqrt((VCdim + np.log(2 / delta)) / n)

		self.srm_penalty = srm_penalty

	def sample_from_D(self, m):
		"""Sample m data samples from D.
		Input: m - an integer, the size of the data sample.

		Returns: np.ndarray of shape (m,2) :
				A two dimensional array of size m that contains the pairs where drawn from the distribution P.
		"""
		res = []
		for i in range(m):
			x = np.random.uniform(0, 1)
			p = -1
			if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
				p = 0.8
			else:
				p = 0.1
			y = np.random.choice([0, 1], p=[1 - p, p])
			res.append((x, y))
		dtype = [('key', float), ('value', int)]
		res = np.array(res, dtype=dtype)
		res = np.sort(res, order='key')
		return res

	def experiment_m_range_erm(self, m_first, m_last, step, k, T):
		"""Runs the ERM algorithm.
		Calculates the empirical error and the true error.
		Plots the average empirical and true errors.
		Input: m_first - an integer, the smallest size of the data sample in the range.
			   m_last - an integer, the largest size of the data sample in the range.
			   step - an integer, the difference between the size of m in each loop.
			   k - an integer, the maximum number of intervals.
			   T - an integer, the number of times the experiment is performed.

		Returns: np.ndarray of shape (n_steps,2).
			A two dimensional array that contains the average empirical error
			and the average true error for each m in the range accordingly.
		"""

		amount = ((m_last - m_first) // step) + 1
		avg_empirical_error = np.array([0 for _ in range(amount)], dtype=float)
		avg_true_error = np.array([0 for _ in range(amount)], dtype=float)
		for i, n in enumerate(range(m_first, m_last + 1, step)):

			dprint(i)

			for t in range(T):
				sample = self.sample_from_D(n)
				erm, empirical_error = intervals.find_best_interval(sample['key'], sample['value'], k)
				true_error = self.true_error(erm)
				avg_empirical_error[i] += empirical_error / n
				avg_true_error[i] += true_error

			avg_empirical_error[i] /= T
			avg_true_error[i] /= T

		"""
		Code to plot relevant graphs:
		
		n_axis = list(range(m_first, m_last + 1, step))
		plt.plot(n_axis, avg_empirical_error, label='Empirical Error')
		plt.plot(n_axis, avg_true_error, label='True Error')
		plt.xlabel('n')
		plt.ylabel('Error')
		plt.legend(loc='upper right')
		plt.show()
		"""

	def experiment_k_range_erm(self, m, k_first, k_last, step):
		"""Finds the best hypothesis for k= 1,2,...,10.
		Plots the empirical and true errors as a function of k.
		Input: m - an integer, the size of the data sample.
			   k_first - an integer, the maximum number of intervals in the first experiment.
			   m_last - an integer, the maximum number of intervals in the last experiment.
			   step - an integer, the difference between the size of k in each experiment.

		Returns: The best k value (an integer) according to the ERM algorithm.
		"""
		k_true_error = []
		k_empirical_error = []

		sample = self.sample_from_D(m)
		for k in range(k_first, k_last + 1, step):
			dprint('k: ' + str(k))
			erm, empirical_error = intervals.find_best_interval(sample['key'], sample['value'], k)
			true_error = self.true_error(erm)
			empirical_error /= m

			k_true_error.append(true_error)
			k_empirical_error.append(empirical_error)

		"""
		Code to plot relevant graphs:

		k_axis = list(range(k_first, k_last + 1, step))
		plt.plot(k_axis, k_empirical_error, label='Empirical Error')
		plt.plot(k_axis, k_true_error, label='True Error')
		plt.xlabel('k')
		plt.ylabel('Error')
		plt.legend(loc='upper right')
		plt.show()
		"""

	def experiment_k_range_srm(self, m, k_first, k_last, step):
		"""Run the experiment in (c).
		Plots additionally the penalty for the best ERM hypothesis.
		and the sum of penalty and empirical error.
		Input: m - an integer, the size of the data sample.
			   k_first - an integer, the maximum number of intervals in the first experiment.
			   m_last - an integer, the maximum number of intervals in the last experiment.
			   step - an integer, the difference between the size of k in each experiment.

		Returns: The best k value (an integer) according to the SRM algorithm.
		"""
		k_true_error = []
		k_empirical_error = []
		k_penalty = []
		k_penalty_empirical_sum = []

		sample = self.sample_from_D(m)
		for k in range(k_first, k_last + 1, step):
			dprint('k: ' + str(k))
			erm, empirical_error = intervals.find_best_interval(sample['key'], sample['value'], k)
			best_k = len(erm)
			penalty = self.srm_penalty(2 * best_k, m)  # VCdim(Hk) = 2 * k
			true_error = self.true_error(erm)
			empirical_error /= m
			penalty_empirical_sum = penalty + empirical_error

			k_true_error.append(true_error)
			k_empirical_error.append(empirical_error)
			k_penalty.append(penalty)
			k_penalty_empirical_sum.append(penalty_empirical_sum)

		"""
		Code to plot relevant graphs:

		k_axis = list(range(k_first, k_last + 1, step))
		plt.plot(k_axis, k_empirical_error, label='Empirical Error')
		plt.plot(k_axis, k_true_error, label='True Error')
		plt.plot(k_axis, k_penalty, label='Penalty')
		plt.plot(k_axis, k_penalty_empirical_sum, label='Penalty and Empirical Sum')
		plt.xlabel('k')
		plt.legend(loc='upper right')
		plt.show()
		"""

	def cross_validation(self, m):
		"""Finds a k that gives a good test error.
		Input: m - an integer, the size of the data sample.

		Returns: The best k value (an integer) found by the cross validation algorithm.
		"""
		train = self.sample_from_D(int(m * 0.8))
		holdout = self.sample_from_D(int(m * 0.2))
		sample = np.concatenate([train, holdout])
		sample = np.sort(sample, order='key')

		error = []
		for k in range(1, 11):
			dprint('k: ' + str(k))
			erm, _ = intervals.find_best_interval(train['key'], train['value'], k)
			empirical_error = self.empirical_error(holdout, erm) / m
			dprint(empirical_error)
			error.append(empirical_error)

		best_k = error.index(min(error)) + 1
		erm, empirical_error = intervals.find_best_interval(sample['key'], sample['value'], best_k)
		empirical_error /= m

		print('best k:', best_k)
		print('the empirical error is:', empirical_error)

		"""
		Code to plot relevant graphs:

		k_axis = list(range(1, 11))
		plt.plot(k_axis, error, label='Error of ERM on holdout')
		plt.xlabel('k')
		plt.ylabel('Error')
		plt.legend(loc='upper right')
		plt.show()
		"""

	def true_error(self, I):
		""" TODO - document
		Computes the true error ep(hI) for the intervals I

		:param I: the intervals. I = [[l1, u1], .. [lk, uk]], 0 <= l1 <= u1 <= l2 <= .. <= lk <= uk <= 1
		:return: int, true error = ep(hI)
		"""
		all = [[0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1]]
		IC = []
		if I[0][0] > 0:
			IC += [[0, I[0][0]]]
		IC += [[I[i][1], I[i + 1][0]] for i in range(len(I) - 1)]
		if I[-1][1] < 1:
			IC += [[I[-1][1], 1]]

		res = 0

		for y in [0, 1]:
			for c in all:
				cl, cu = c[0], c[1]
				for interval in I + IC:
					hx = 1 if interval in I else 0
					l, u = interval[0], interval[1]
					if cl <= l <= cu:
						if u <= cu:  # [cl, {l, u}, cu]
							x = (l + u) / 2
							length = u - l
							res += length * self.p(y, x) * self.zo(hx, y)
						else:  # u > cu --> [cl, {l, cu}, u]
							x = (l + cu) / 2
							res += (cu - l) * self.p(y, x) * self.zo(hx, y)
					elif l < cl:
						if cl < u:  # [l, cl, u, cu] or [l, cl, cu, u] --> {cl, min(u, cu)}
							end = min(cu, u)
							x = (cl + end) / 2
							res += (end - cl) * self.p(y, x) * self.zo(hx, y)
		return res

	def empirical_error(self, train, I):
		""" TODO - document

		:param S:
		:param I:
		:return:
		"""
		error = 0
		for dot in train:
			x, y = dot['key'], dot['value']
			hx = 0
			for interval in I:
				if interval[0] <= x <= interval[1]:
					hx = 1
					break
			if hx != y:
				error += 1
		return error


if __name__ == '__main__':
	ass = Assignment2()
	# ass.experiment_m_range_erm(10, 100, 5, 3, 100)
	# ass.experiment_k_range_erm(1500, 1, 10, 1)
	# ass.experiment_k_range_srm(1500, 1, 10, 1)
	# ass.cross_validation(1500)
