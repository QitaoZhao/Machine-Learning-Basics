import numpy as np
import matplotlib.pyplot as plt


def MF(critics_map, epoch, k, lr, weight_decay):
	U, M = critics_map.shape
	user = np.random.normal(0, 1, (U, k))
	movie = np.random.normal(0, 1, (k, M))
	mask = critics_map == 0
	loss_history = []

	for i in range(epoch):
		temp = critics_map - user.dot(movie)
		temp[mask] = 0
		loss = 0.5 * np.linalg.norm(temp) ** 2 
		loss += 0.5 * weight_decay * (np.linalg.norm(user)**2 + np.linalg.norm(user)**2)

		grad_u = -temp.dot(movie.T) + weight_decay * user
		grad_m = -user.T.dot(temp) + weight_decay * movie

		user -= lr * grad_u
		movie -= lr * grad_m

		if i % 1000 == 0:
			loss_history.append(loss)
			print("Epoch: %d, loss: %.4f" % (i, loss))

			if i != 0:
				if abs(loss_history[-1] - loss_history[-2]) < 1e-6:
					break

	return user.dot(movie)


np.random.seed()
users = ['Lisa Rose', 'Gene Seymour', 'Michael Phillips', 'Claudia Puig', 
		 'Mick LaSalle', 'Jack Matthews', 'Toby']

movies = ["Lady in the Water", "Snakes on a Plane", "Just My Luck", 
		  "Superman Returns", "You, Me and Dupree", "The Night Listener"]

critics = {'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
'The Night Listener': 3.0},
'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
'Just My Luck': 1.5, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5,
'The Night Listener': 3.0},
'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
'Superman Returns': 3.5, 'The Night Listener': 4.0},
'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
'Superman Returns': 4.0, 'You, Me and Dupree': 2.5, 'The Night Listener': 4.5},
'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
'Just My Luck': 2.0, 'Superman Returns': 3.0, 'You, Me and Dupree': 2.0,
'The Night Listener': 3.0},
'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
'Superman Returns': 5.0, 'You, Me and Dupree': 3.5, 'The Night Listener': 3.0,},
'Toby': {'Snakes on a Plane': 4.5, 'Superman Returns': 4.0, 'You, Me and Dupree': 1.0,}}

# Pre-processing
critics_map = np.zeros((7, 6))

for i in range(7):
	for j in range(6):
		critics_map[i, j] = critics[users[i]].get(movies[j], 0)

mask = critics_map != 0

epoch = 200000
k = 10
learning_rate = 0.05
weight_decay = 0.01
predicted_critics_map = MF(critics_map, epoch, k, learning_rate, weight_decay)
print(critics_map)
print(predicted_critics_map)
predicted_critics_map[mask] = 0
print("'%s' would be recommended for Toby.\nIt got %0.2f!" % (movies[np.argmax(predicted_critics_map[-1])], np.max(predicted_critics_map[-1])))
