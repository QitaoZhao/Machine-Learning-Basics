import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt


def knn(X_train, y_train, X_val, y_val, k, set_mode):
	num_val, D = X_val.shape
	predicted_label = np.ndarray((num_val, ), dtype='S32')
	for i in range(num_val):
		L2 = np.sqrt(np.sum((X_train - X_val[i, :]) ** 2, axis=1))
		predicted_label[i] = mode(y_train[np.argsort(L2, axis=0)][:k])[0][0]
		predicted_label[i] = predicted_label[i].decode('utf-8')
	accuracy = np.sum(predicted_label==y_val) / num_val
	print("k = %d, %s accuracy: %.2f" % (k, set_mode, accuracy))
	return accuracy


# Extract data set
data = np.ndarray((150, 4), dtype=float)
label = np.ndarray((150, ), dtype='S32')
file = open('./iris.txt')
org = file.readlines()
for i, line in enumerate(org):
	data[i, :4] = np.array(line[:-1].split(',')[:4], dtype=float)
	label[i] = line[:-1].split(',')[4]

# Split the data set into train_set and val_set
np.random.seed(231)
random_list = np.arange(150)
np.random.shuffle(random_list)
data = data[random_list, :]
label = label[random_list]
X_train = data[:120, :]
y_train = label[:120]
X_val = data[120:, :]
y_val = label[120:]

# Train & validation
x = np.linspace(1, 150, 31).astype(int)
train_accuracy = []
val_accuracy = []
for k in x:
	train_accuracy.append(knn(X_train, y_train, X_train, y_train, k, 'train'))
	val_accuracy.append(knn(X_train, y_train, X_val, y_val, k, 'validation'))

# Plot the result
plt.plot(x, train_accuracy, label="Train_accuracy")
plt.plot(x, val_accuracy, label="Validation_accuracy")
plt.title('K-nearest neighbors')
plt.xticks(x, rotation=60)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend(loc="best", ncol=4)
plt.grid(linestyle='--', linewidth=0.5)
plt.show()