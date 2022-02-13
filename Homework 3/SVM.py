import numpy as np
import matplotlib.pyplot as plt


def generate_data(dot_num):
	mean = (0, 0)
	cov = [[1, 0], [0, 2]]
	X_A = np.random.multivariate_normal(mean, cov, dot_num)
	X_B = np.random.uniform([-5, -5], [5, 5], [dot_num, 2])	
	X = np.concatenate((X_A, X_B), axis=0)
	y = np.zeros(2*dot_num)
	y[:dot_num] -= 1
	y[dot_num:] += 1
	return X, y

def linear_kernel(X_1, X_2):
	N1, D = X_1.shape
	if len(X_2.shape) == 1:
		N2 = 1
	else:
		N2, D = X_2.shape
	G = np.ones((N1, N2, D))
	G = G * X_2
	G = (G.transpose(1,0,2) * X_1).transpose(1, 0, 2)
	return G.sum(axis=2)

def gaussian_kernel(X_1, X_2):
	N1, D = X_1.shape
	if len(X_2.shape) == 1:
		N2 = 1
	else:
		N2, D = X_2.shape
	G = np.zeros((N1, N2, D))
	G = (G.transpose(1,0,2) + X_1).transpose(1, 0, 2)
	G -= X_2
	G = np.linalg.norm(G, axis=2) ** 2
	return np.exp(-G/(2 * sigma**2))

def rectifier(x, C):
	return np.maximum(np.minimum(x, C), 0)

def train(X_train, y_train, epoch, alpha, learning_rate):
	N, D = X_train.shape
	G = gaussian_kernel(X_train, X_train)
	loss_history, accuracy_history = [], []

	for i in range(epoch): 
		temp = (G * alpha * y_train).T * y_train
		temp = temp.T
		grad = temp.sum(axis=1) - 1 
		loss = -alpha.sum() + (0.5 * temp.T * alpha).T.sum()
		loss_history.append(loss)

		if i % 100 == 0:
			y = np.zeros(N)
			for j in range(N):
				G_temp = gaussian_kernel(X_train, X_train[j, :])
				score = ((G_temp.T * alpha * y_train).T).sum(axis=0)
				y[j] = 1 if score >= 0 else -1
			correct = (y == y_train).sum()
			accuracy_history.append(correct/N)
			print("Epoch: %d, loss: %.4f, training accuracy: %.2f%% (%d/%d)" % (i, loss, correct/N*100, correct, N))

		alpha -= learning_rate * grad
		alpha = rectifier(alpha, C)
	
	print("Epoch: %d, loss: %.4f, training accuracy: %.2f%% (%d/%d)" % (i, loss, correct/N*100, correct, N))

	return alpha

def test(X_train, y_train, X_test, y_test, alpha):
	N, D = X_train.shape
	y = np.zeros(N) 
	for i in range(N):
		G = gaussian_kernel(X_train, X_test[i, :])
		score = ((G.T * alpha * y_train).T).sum()
		y[i] = 1 if score >= 0 else -1
	correct = (y == y_test).sum()
	print("Test accuracy: %.2f%% (%d/%d)" % (correct/N*100, correct, N))
	return y, correct/N

def func(X, Y, alpha):
	scores = np.zeros_like(X)
	for i in range(100):
		for j in range(100):
			temp = np.array([X[i, j], Y[i, j]])
			G = gaussian_kernel(X_train, temp)
			scores[i, j] = ((G.T * alpha * y_train).T).sum()
	return scores


np.random.seed(2021)

dot_num = 100
C = 5.134395 # np.random.uniform(10)
epoch = 1000
alpha = np.random.uniform(0, C, (2*dot_num, ))
learning_rate = 1e-2
sigma = 0.82836 # 10 ** np.random.uniform(-2, 0)

# Generate data points
X_train, y_train = generate_data(dot_num)
X_test, y_test = generate_data(dot_num)

plt.figure(figsize=(8.5,10))

plt.subplot(3, 1, 1)
plt.scatter(X_test[:dot_num, 0], X_test[:dot_num, 1], s=15, c='b', label="A")
plt.scatter(X_test[dot_num:, 0], X_test[dot_num:, 1], s=15, c='r', label="B")
plt.title("Original distribution")
plt.legend(loc="best", ncol=4)

alpha_1000 = train(X_train, y_train, epoch, alpha, learning_rate)

predicted_y, accuracy_0 = test(X_train, y_train, X_test, y_test, alpha)
mask_A = np.where(predicted_y==-1)
mask_B = np.where(predicted_y==1)

predicted_y_1000, accuracy_1000 = test(X_train, y_train, X_test, y_test, alpha_1000)
mask_A_1000 = np.where(predicted_y_1000==-1)
mask_B_1000 = np.where(predicted_y_1000==1)

x = np.linspace(-5, 5, dot_num)
y = np.linspace(-5, 5, dot_num)
X, Y = np.meshgrid(x, y)

plt.subplot(3, 1, 2)
plt.contourf(X, Y, func(X, Y, alpha))
plt.scatter(X_test[mask_A, 0], X_test[mask_A, 1], s=15, c='b', label="A")
plt.scatter(X_test[mask_B, 0], X_test[mask_B, 1], s=15, c='r', label="B")
plt.title("Predicted distribution before training, test accuracy: " + str(100*accuracy_0) + "%")
plt.legend(loc="best", ncol=4)

plt.subplot(3, 1, 3)
plt.contourf(X, Y, func(X, Y, alpha_1000))
plt.scatter(X_test[mask_A_1000, 0], X_test[mask_A_1000, 1], s=15, c='b', label="A")
plt.scatter(X_test[mask_B_1000, 0], X_test[mask_B_1000, 1], s=15, c='r', label="B")
plt.title("Predicted distribution after training, test accuracy: " + str(100*accuracy_1000) + "%")
plt.legend(loc="best", ncol=4)
plt.subplots_adjust(hspace=0.35, top=0.95)
plt.show()