import numpy as np
import matplotlib.pyplot as plt


def generate_data(dot_num):
	mean_A = (0, 0)
	cov_A = [[1, 0], [0, 1]]
	mean_B = (1, 2)
	cov_B = [[1, 0], [0, 2]]
	X_A = np.random.multivariate_normal(mean_A, cov_A, dot_num)
	X_B = np.random.multivariate_normal(mean_B, cov_B, dot_num)
	X = np.ones((2*dot_num, 3))
	X[:, 1:] = np.concatenate((X_A, X_B), axis=0)
	y = np.zeros(2*dot_num)
	y[:dot_num] += 1
	y[dot_num:] -= 1
	return X, y

def train(X_train, y_train, epoch, learning_rate):
	N, D = X_train.shape
	theta = np.zeros(3)
	y_prediected = np.zeros_like(y_train)

	for i in range(epoch): 

		scores = X_train.dot(theta)
		y_prediected[scores>=0] = 1
		y_prediected[scores<0] = -1
		correct = (y_prediected==y_train).sum()

		if i % 100 == 0:
			learning_rate /= 2
		if i % 1000 == 0:
			print("Epoch: %d, accuracy: %.4f (%d/%d)" % (i, correct/N, correct, N))

		for j in range(N):
			temp = theta.dot(X_train[j, :])
			y = 1 if temp >=0 else -1
			if y != y_train[j]:
				theta += y_train[j] * X_train[j] * learning_rate

	print("Epoch: %d, accuracy: %.4f (%d/%d)" % (i, correct/N, correct, N))

	return theta

def test(X_test, y_test, theta):
	N, D = X_train.shape
	y_prediected = np.zeros(N) 
	scores = X_test.dot(theta)
	y_prediected[scores>=0] = 1
	y_prediected[scores<0] = -1
	correct = (y_prediected == y_test).sum()
	print("Test accuracy: %.2f%% (%d/%d)" % (correct/N*100, correct, N))


np.random.seed(2021)

dot_num = 30
epoch = 10000
learning_rate = 2

# Generate data points
index = np.arange(2*dot_num)
np.random.shuffle(index)
X_train, y_train = generate_data(dot_num)
X_train, y_train = X_train[index, :], y_train[index]
X_test, y_test = generate_data(dot_num)
theta = train(X_train, y_train, epoch, learning_rate)
predicted_y = test(X_test, y_test, theta)

plt.scatter(X_test[:dot_num, 1], X_test[:dot_num, 2], c='b', label="A")
plt.scatter(X_test[dot_num:, 1], X_test[dot_num:, 2], c='r', label="B")
plt.plot(X_test[:, 1], -(X_test[:, 1]*theta[1]+theta[0])/theta[2], color="darkorange", label="Decision boundary")
plt.title("Perceptron")
plt.legend(loc="best", ncol=4)
plt.show()