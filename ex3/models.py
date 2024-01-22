import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class Perceptron:
	def fit(self, X, y):
		X = np.hstack((np.ones((y.size,1)), X))
		w = np.zeros(X.shape[1])
		while True:
			negs = np.nonzero((X @ w) * y <= 0)[0]
			if negs.size == 0:
				break
			i = negs[0]
			w += X[i, :] * y[i]
		self.model = w

	def predict(self, X):
		return np.sign(np.hstack((np.ones((X.shape[0],1)), X)) @ self.model)  # adding intercept

	def score(self, X, y):
		y_hat = self.predict(X)
		return calc_stats(y, y_hat)


class LDA:

	def fit(self, X, y):
		split = y == 1
		probab = y[split].sum() / y.size
		X1 = X[split, :]
		Xmin1 = X[~split, :]
		mu = (np.mean(X1, axis=0), np.mean(Xmin1, axis=0))
		Sig_inv = np.linalg.inv(((X1.T @ X1) + (Xmin1.T @ Xmin1)) / (y.size - 1))
		delta_1 = lambda x: x @ Sig_inv @ mu[0] - .5 * mu[0].T @ Sig_inv @ mu[0] + np.log(probab)
		delta_min1 = lambda x: x @ Sig_inv @ mu[1] - .5 * mu[1].T @ Sig_inv @ mu[1] + np.log(probab)
		self.model = (delta_1, delta_min1)

	def predict(self, X):
		res1 = self.model[0](X)
		res_min1 = self.model[1](X)
		res = np.argmax(np.vstack((res_min1, res1)), axis=0)
		res[res == 0] = -1  # if delta_-1 was bigger than its index (0) appears
		return res
		
	def score(self, X, y):
		y_hat = self.predict(X)
		return calc_stats(y, y_hat)


class SVM:
	def __init__(self):
		self.model = SVC(C=1e10, kernel='linear')

	def fit(self, X, y):
		self.model.fit(X, y)

	def predict(self, X):
		return self.model.predict(X)

	def score(self, X, y):
		y_hat = self.predict(X)
		return calc_stats(y, y_hat)


class Logistic:

	def __init__(self):
		self.model = LogisticRegression(solver='liblinear')

	def fit(self, X, y):
		self.model.fit(X, y)

	def predict(self, X):
		return self.model.predict(X)

	def score(self, X, y):
		y_hat = self.predict(X)
		return calc_stats(y, y_hat)


class DecisionTree:

	def __init__(self):
		self.model = DecisionTreeClassifier(max_depth=12)

	def fit(self, X, y):
		self.model.fit(X, y)

	def predict(self, X):
		return self.model.predict(X)

	def score(self, X, y):
		y_hat = self.predict(X)
		return calc_stats(y, y_hat)


def calc_stats(y, y_hat):
	n_samp = y.size
	score = {'num_samples': n_samp}
	P = (y_hat == 1).sum()
	N = (y_hat == -1).sum()
	FP = np.logical_and(y == -1, y_hat == 1).sum()
	TP = np.logical_and(y == 1, y_hat == 1).sum()
	FN = np.logical_and(y == 1, y_hat == -1).sum()
	TN = np.logical_and(y == -1, y_hat == -1).sum()
	score['error'] = (FP + FN) / n_samp
	score['accuracy'] = (TP + TN) / n_samp
	score['FPR'] = FP / N
	score['TPR'] = TP / P
	score['precision'] = TP / (TP + FP)
	score['specificity'] = TN / N
	return score
