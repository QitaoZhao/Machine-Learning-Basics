from text_reader import Book
import numpy as np


def stat(author, indices, verbose=True):
	book = Book()
	book.author = author
	for i in indices:
		book.read(r"/Users/zhaoqitao/Desktop/assignment1/text/%s/%d.txt" % (author, i))
		book.countWords()
	if verbose:
		print(author + ":")
		book.show(10)
	return(book)

def MLE(X_train, y_train, X_val, y_val, margin, mode, indices=None, verbose=True):
	num_train, D = X_train.shape
	num_val, D = X_val.shape
	predicted_label = np.ndarray((num_val, ), dtype='S32')

	indices_H = np.where(y_train==b'hamilton')[0]
	indices_M = np.where(y_train==b'madison')[0]

	probs_integrated_H = np.minimum(np.maximum(X_train[indices_H, :].mean(axis=0), margin), 1-margin)
	probs_integrated_M = np.minimum(np.maximum(X_train[indices_M, :].mean(axis=0), margin), 1-margin)

	for i in range(num_val):
		mask_0 = X_val[i, ] == 0
		mask_1 = X_val[i, ] == 1
		log_prob_H = (np.log(probs_integrated_H[mask_1]) * X_val[i, mask_1]).sum() 
		log_prob_H += (np.log(1-probs_integrated_H[mask_0]) * (1-X_val[i, mask_0])).sum()
		log_prob_M = (np.log(probs_integrated_M[mask_1]) * X_val[i, mask_1]).sum() 
		log_prob_M += (np.log(1-probs_integrated_M[mask_0]) * (1-X_val[i, mask_0])).sum()
		if log_prob_H > log_prob_M:
			predicted_label[i] = 'hamilton'
		elif log_prob_H < log_prob_M:
			predicted_label[i] = 'madison'
		else:
			print("Value error!")
	if mode == 'val':
		accuracy = np.sum(predicted_label==y_val) / num_val
		if verbose:
			print("Validation accuracy: {:.2f}%".format(accuracy*100))
		return accuracy
	else:
		print("Predict with MLE:")
		for i in range(len(indices)):
			print("The no.%d article is most likely written by %s" % (indices[i], predicted_label[i].decode('utf-8').capitalize()))

def crossValidation(X, y, num_folds, margin, k=None):
	print("Cross-validation:")
	X_train_folds = np.array_split(X, num_folds)
	y_train_folds = np.array_split(y, num_folds)
	accuracies = []

	for i in range(num_folds):
		X, y = X_train_folds[:], y_train_folds[:]
		X.pop(i)
		y.pop(i)
		X_train = X_train_folds[i]
		y_train = y_train_folds[i]
		X_val = np.concatenate(X, axis=0)
		y_val = np.concatenate(y, axis=0)
		accuracies.append(MLE(X_train, y_train, X_val, y_val, margin, 'val', verbose=True))

	print("Mean validation accuracy: {:.2f}% \n".format(sum(accuracies)/num_folds*100))

	return sum(accuracies)/num_folds


hamilton = [1, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
			30, 31, 32, 33, 34, 35, 36, 59, 60, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 
			74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85]
madison = [10, 14, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 58]
unknown = [49, 50, 51, 52, 53, 54, 55, 56, 57, 62, 63]

##################### Extract data #####################

H = stat('hamilton', hamilton, False)
M = stat('madison', madison, False)

vocabulary = []
for k, v in H.stat.items():
	if v >= 5 and k not in vocabulary:
		vocabulary.append(k)
for k, v in M.stat.items():
	if v >= 5 and k not in vocabulary:
		vocabulary.append(k)

num_books = len(hamilton) + len(madison)
len_vocabulary = len(vocabulary)
data = np.ndarray((num_books, len_vocabulary), dtype=np.float32)
label = np.ndarray((num_books, ), dtype='S32')
X_test = np.ndarray((len(unknown), len_vocabulary), dtype=np.float32)

for i in range(len(hamilton)):
	temp = stat('hamilton', [hamilton[i]], False)
	num_occur = np.zeros((len_vocabulary, ))
	for j in range(len(vocabulary)):
		num_occur[j] = vocabulary[j] in temp.stat 
	data[i] = num_occur
	label[i] = temp.author

for i in range(len(madison)):
	temp = stat('madison', [madison[i]], False)
	num_occur = np.zeros((len_vocabulary, ))
	for j in range(len(vocabulary)):
		num_occur[j] = vocabulary[j] in temp.stat
	data[i+len(hamilton)] = num_occur
	label[i+len(hamilton)] = temp.author

for i in range(len(unknown)):
	temp = stat('unknown', [unknown[i]], False)
	num_occur = np.zeros((len_vocabulary, ))
	for j in range(len(vocabulary)):
		num_occur[j] = vocabulary[j] in temp.stat
	X_test[i] = num_occur

np.random.seed(2021)

# Randomize data and labels
random_list = np.arange(num_books)
np.random.shuffle(random_list)
data = data[random_list, :].astype(np.int32)
label = label[random_list]

# Cross-validation
# margin = 10 ** np.random.uniform(-3, -1)
margin = 0.05
crossValidation(data, label, 3, margin, 'MLE')

# Predict
MLE(data, label, X_test, None, margin, 'test', unknown)
