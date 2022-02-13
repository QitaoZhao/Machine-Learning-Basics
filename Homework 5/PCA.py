import numpy as np
import matplotlib.pyplot as plt


def generate_data(dot_num):
	mean = [2, 1]
	covariance = [[0.5, 0.3], [0.3, 1]]
	return np.random.multivariate_normal(mean, covariance, dot_num)

def PCA(X):
	N, D = X.shape
	mean = np.mean(X, axis=0)
	variance = ((X - mean)**2).mean(axis=0)

	# Standarize the data
	X_centered = (X - mean) / variance

	covariance = X.T.dot(X) / N
	w, v = np.linalg.eig(covariance) 
	loss = np.min(w)
	return v[:, np.argmax(w)], v[:, np.argmin(w)], X_centered, loss # , loss


np.random.seed()
X = generate_data(50)
u_1, u_2, X_centered, loss = PCA(X)
x_1 = np.arange(-3, 4)
x_2 = np.linspace(-0.5, 0.5)

# Plot the result
plt.figure(figsize=(6, 6))
plt.scatter(X_centered[:, 0], X_centered[:, 1])
plt.plot(x_1, x_1*u_1[1]/u_1[0], c="royalblue", label="1st component")
plt.plot(x_2, x_2*u_2[1]/u_2[0], c="cornflowerblue", label="2nd component")
plt.title("PCA, reconstruction error: %.3f" % loss)
plt.legend(loc="best", ncol=4)
plt.xlim(-4, 4)
plt.ylim(-3, 3)
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box') # Set x-axis and y-axis to an equal scale
plt.show()