import numpy as np
import matplotlib.pyplot as plt

def generate_dots(dot_num):
    mean_A = (0, 0)
    cov_A = [[1, 0], [0, 1]]
    mean_B = (1, 2)
    cov_B = [[1, -0.3], [-0.3, 2]]
    A = np.random.multivariate_normal(mean_A, cov_A, dot_num)
    B = np.random.multivariate_normal(mean_B, cov_B, dot_num)    
    X_A = np.ones((dot_num, 3))
    X_B = np.ones((dot_num, 3))
    X_A[:, 1:3], X_B[:, 1:3] = A, B
    X = np.concatenate((X_A, X_B), axis=0)
    return X

def logistic(W, X):
    return 1 / (1+np.exp(X.dot(W)))

def GD(X, y, mode, learning_rate, print_every=200):
    N, D = X.shape
    W = np.random.rand(3)
    t = 0
    log_likelihood = y * (X.dot(W)) - np.log(1 + np.exp(X.dot(W)))
    log_likelihood = log_likelihood.sum()
    log_likelihood_history = [log_likelihood]
    while True:
        if t % print_every == 0:
            print("Epoch: %d, log_likelihood: %.4f" % (t, log_likelihood))
        grad = y - np.exp(X.dot(W))/(1 + np.exp(X.dot(W)))
        grad = ((X.T * grad).T)
        if mode == 'SGD':
            grad = grad[np.random.randint(N), :]
        else:
            grad = grad.mean(axis=0)
        W += learning_rate * grad
        log_likelihood = y * X.dot(W) - np.log(1 + np.exp(X.dot(W)))
        log_likelihood = log_likelihood.sum()
        if abs(log_likelihood - log_likelihood_history[-1]) < 1e-6:
            break
        log_likelihood_history.append(log_likelihood)
        t += 1
    print("Epoch: %d, log_likelihood: %.4f" % (t, log_likelihood))
    return W

def test(W, mode):
    X = generate_dots(dot_num)
    y_A = np.zeros((dot_num, ))
    y_B = np.ones((dot_num, ))
    y = np.concatenate((y_A, y_B), axis=0)
    plt.scatter(X[:50, 1], X[:50, 2], c='b', label="A")
    plt.scatter(X[50:, 1], X[50:, 2], c='r', label="B")
    plt.plot(X[:, 1], (-W[1]*X[:, 1]-W[0])/W[2], color="darkorange", label="Decision boundary")
    plt.title("Logistic regression with " + mode)
    plt.legend(loc="best", ncol=4)
    plt.show()


dot_num = 50
learning_rate = 5e-2
np.random.seed(231)
X = generate_dots(dot_num)
y_A = np.zeros((dot_num, ))
y_B = np.ones((dot_num, ))
y = np.concatenate((y_A, y_B), axis=0)

# Do 'GD' or 'SGD'
# W = GD(X, y, 'GD', learning_rate)
W = GD(X, y, 'SGD', learning_rate)
test(W, 'SGD')