import numpy as np
from matplotlib import pyplot as plt
import math

"""
DESC:   This is the code relevant to question 4 in the "Decision Rules and Concentration Bounds" section
"""

n = 20
N = 200_000

# Build the matrix and sum on rows
matrix = np.array([[np.random.randint(0, 2) for _ in range(n)] for _ in range(N)])
rows = (1 / n) * np.sum(matrix, axis=1)

# Create epsilons list
eps = np.linspace(0, 1, 50)

# Compute empirical
empirical = []
for e in eps:
	s = 0
	for r in rows:
		if abs(r - 1 / 2) > e:
			s += 1
	empirical.append(s / N)

def h(epsilon):
	return 2 * np.exp(-np.power(epsilon, 2) * n / 2)


# Compute hoeffding bound
hoeffding = [h(e) for e in eps]

# Plot both graphs
plt.plot(eps, empirical, label="empirical")
plt.plot(eps, hoeffding, label="hoeffding")
plt.legend(loc="upper right")
plt.show()
