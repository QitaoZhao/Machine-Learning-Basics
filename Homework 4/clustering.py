import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def generate_data():
	X_A = np.random.multivariate_normal((0,0), [[1,0],[0,1]], 30)
	X_B = np.random.multivariate_normal((1,2), [[2,0],[0,1]], 20)
	X_C = np.random.multivariate_normal((2,0), [[1,0.3],[0.3,1]], 20)
	X = np.concatenate((X_A, X_B, X_C), axis=0)
	y = np.ones(30 + 20 + 20)
	y[30:] += 1
	y[50:] += -2
	return X, y

def check_stability(k, X, means):
	N, D = X.shape
	L2 = np.zeros((k, N, D))
	L2 += X
	L2 = np.sqrt(((L2.transpose(1,0,2) - means ) ** 2).sum(axis=2)) # (N, K)

	for i in range(N):
		for j in range(k):
			if L2[i, j] < 1e-3:
				means[j] += np.random.normal(0, 0.1, size=(D))

	return means

def k_means(k, X, epoch):
	N, D = X.shape
	rng = np.random.default_rng(2050)
	centers = rng.choice(X, size=k, replace=False, p=None, axis=0)
	y = np.zeros(N)
	loss_history = []
	print("K-means:")

	for i in range(epoch):
		distance_matrix = np.zeros((N, k, D))
		distance_matrix = (distance_matrix.transpose(1,0,2) + X).transpose(1,0,2)
		distance_matrix -= centers
		distance_matrix = np.sqrt((distance_matrix ** 2).sum(axis=2))
		y = np.argmin(distance_matrix, axis=1)
		loss = distance_matrix[np.arange(N), y].mean()

		if len(loss_history) > 0:
			if abs(loss-loss_history[-1]) < 1e-3:
				break

		loss_history.append(loss)

		for j in range(k):
			mask = np.where(y==j)[0]
			centers[j, :] = np.mean(X[mask, :], axis=0)

		if i % 1 == 0:
			print(" Epoch: %d, loss: %.4f" % (i, loss))

	return y, centers

def EM(k, X, epoch):
	N, D = X.shape
	rng = np.random.default_rng(2050)
	means = rng.choice(X, size=k, replace=False, p=None, axis=0)
	covars = np.identity(D)
	mixing_coeffs = np.zeros(k) + 1/k # start uniformly
	mul_normals = []
	log_likelihoods = []
	print("EM algorithm:")

	temp = np.zeros((N, k))
	for i in range(k):
		mul_normals.append(multivariate_normal(means[i], covars))
		temp[:, i] = mul_normals[i].pdf(X)

	for i in range(epoch):

		# E step
		temp *= mixing_coeffs
		log_likelihood = (np.log((temp).sum(axis=1))).sum()
		log_likelihoods.append(log_likelihood)
		gamma = (temp.T / temp.sum(axis=1)).T # (N, k)
		N_k = gamma.sum(axis=0) # (k, )

		# M step
		means = np.ones((N, k, D))
		means = (means.transpose(1,0,2) * X).transpose(2,1,0) * gamma / N_k # (D, N, k)
		means = means.transpose(2,0,1).sum(axis=2)

		covars = np.zeros((k, D, D))
		for j in range(k):
			covars[j, :, :] = ((X-means[j]).T * gamma[:, j]).dot(X-means[j]) / N_k[j] 

		mixing_coeffs = N_k / N

		# Evaluate the log likelihood
		for j in range(k):
			mul_normals[j] = multivariate_normal(means[j], covars[j])
			temp[:, j] = mul_normals[j].pdf(X)

		if i % 1 == 0:
			print(" Epoch: %d, log_likelihood: %.4f" % (i, log_likelihood))

		if i > 0:
			if abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1e-3:
				break

	return gamma


np.random.seed(2050) # 2050

X, y = generate_data()
y_predicted_1, _ = k_means(3, X, 10)
log_likelihood = EM(3, X, 10)
y_predicted_2 = np.argmax(log_likelihood, axis=1)

plt.figure(figsize=(7, 9))

plt.subplot(3, 1, 1)
plt.scatter(X[np.where(y==0)[0], 0], X[np.where(y==0)[0], 1], c='r', label="A")
plt.scatter(X[np.where(y==1)[0], 0], X[np.where(y==1)[0], 1], c='g', label="B")
plt.scatter(X[np.where(y==2)[0], 0], X[np.where(y==2)[0], 1], c='b', label="C")
plt.title("Original distribution")
plt.legend(loc="best", ncol=4)
plt.ylim(-2, 4.5)

plt.subplot(3, 1, 2)
plt.scatter(X[np.where(y_predicted_1==0)[0], 0], X[np.where(y_predicted_1==0)[0], 1], c='r', label="A")
plt.scatter(X[np.where(y_predicted_1==1)[0], 0], X[np.where(y_predicted_1==1)[0], 1], c='g', label="B")
plt.scatter(X[np.where(y_predicted_1==2)[0], 0], X[np.where(y_predicted_1==2)[0], 1], c='b', label="C")
plt.title("K-means, accuracy: %.2f%%" % (np.sum(y_predicted_1==y)/70*100))
plt.legend(loc="best", ncol=4)
plt.ylim(-2, 4.5)

plt.subplot(3, 1, 3)
plt.scatter(X[np.where(y_predicted_2==0)[0], 0], X[np.where(y_predicted_2==0)[0], 1], c='r', label="A")
plt.scatter(X[np.where(y_predicted_2==1)[0], 0], X[np.where(y_predicted_2==1)[0], 1], c='g', label="B")
plt.scatter(X[np.where(y_predicted_2==2)[0], 0], X[np.where(y_predicted_2==2)[0], 1], c='b', label="C")
plt.title("EM algorithm, accuracy: %.2f%%" % (np.sum(y_predicted_2==y)/70*100))
plt.legend(loc="best", ncol=4)
plt.ylim(-2, 4.5)

plt.subplots_adjust(hspace=0.25, top=0.95, bottom=0.05)
plt.show()

