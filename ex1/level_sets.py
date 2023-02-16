import numpy as np
import matplotlib.pyplot as plt

"""
DESC:   This is the code relevant to question 2 in the "Decision Rules and Concentration Bounds" section
"""

# Define functions
f0 = lambda x: np.exp(-0.5 * (np.power(x[0], 2) + 0.5 * np.power(x[1], 2))) / (2 * np.pi * np.sqrt(2))
f1 = lambda x: np.exp(-0.5 * (np.power((x[0] - 1), 2) + 0.5 * (np.power((x[1] - 1), 2)))) / (2 * np.pi * np.sqrt(2))
bound = lambda x, p: 2 * np.log((1 - p) / p) + 3 / 2 - 2 * x[0] - x[1]

# Create function f0
x1 = np.linspace(-3, 4)
x2 = np.linspace(-3, 4)
x1, x2 = np.meshgrid(x1, x2)
zx = f0((x1, x2))

# Create function f1
y1 = np.linspace(-3, 4)
y2 = np.linspace(-3, 4)
y1, y2 = np.meshgrid(y1, y2)
zy = f1((y1, y2))

# Create decision bounds
b1 = np.linspace(-3, 4)
b2 = np.linspace(-3, 4)
b1, b2 = np.meshgrid(b1, b2)

zb1 = bound((b1, b2), 0.25)
zb2 = bound((b1, b2), 0.5)
zb3 = bound((b1, b2), 0.99)

# Plot level sets of both functions
a = plt.contour(x1, x2, zx, colors='green')
b = plt.contour(y1, y2, zy, colors='red')

# Plot decision bounds - created as the level set of c = 0.
c = plt.contour(b1, b2, zb1, colors="purple", levels=[0])
d = plt.contour(b1, b2, zb2, colors="black", levels=[0])
e = plt.contour(b1, b2, zb3, colors="pink", levels=[0])

# Add labels
a.collections[0].set_label("f0")
b.collections[0].set_label("f1")
c.collections[0].set_label("p = 0.25")
d.collections[0].set_label("p = 0.5")
e.collections[0].set_label("p = 0.99")

plt.legend(loc="upper left")
plt.show()
