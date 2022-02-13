import numpy as np
import matplotlib.pyplot as plt

def normal_equations(X, y):
    W = np.zeros((2, ))
    W = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
    plt.subplot(2, 1, 1)
    plt.scatter(X[:, 1], y, c='r')
    plt.plot(X[:, 1], X[:, 1] * W[1] + W[0], color="dodgerblue", label="Normal_equations")
    plt.title('Linear regression with normal equations')
    plt.legend(loc="best", ncol=4)
    return W

def GD(X, y, learning_rate, print_every=500):
    W = np.random.rand(2)
    t = 0
    loss = 0.5 * ((y - X.dot(W)) ** 2)
    loss = loss.mean()
    loss_history = [loss]

    while True:
        if t % print_every == 0:
            print("Iteration: %d, loss: %.4f" % (t, loss))
        grad = -((y - X.dot(W)) * (X.T)).T
        grad = grad.mean(axis=0)
        W -= learning_rate * grad
        loss = 0.5 * ((y - X.dot(W)) ** 2)
        loss = loss.mean()
        if abs(loss - loss_history[-1]) < 1e-6:
            print("Iteration: %d, loss: %.4f" % (t, loss))
            break
        loss_history.append(loss)
        t += 1

    plt.subplot(2, 1, 2)
    plt.scatter(X[:, 1], y, c='r')
    plt.plot(X[:, 1], X[:, 1] * W[1] + W[0], color="darkorange", label="GD")
    plt.title('Linear regression with gradient descent')
    plt.legend(loc="best", ncol=4)
    return W

mean = (1,2)
cov = [[1, -0.3], [-0.3, 2]]
dot_num = 30
learning_rate = 1e-3
np.random.seed(231)
dots = np.random.multivariate_normal(mean, cov, dot_num)
X = np.ones((dot_num, 2))
X[:, 1], y = dots[:, 0], dots[:, 1]

normal_equations(X, y)
GD(X, y, 5e-3)
plt.subplots_adjust(hspace=0.5)
plt.show()
