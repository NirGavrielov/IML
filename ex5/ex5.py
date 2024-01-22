import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def gen_data():
	m = 1500
	X = np.random.uniform(-3.2, 2.2, m)
	err = np.random.normal(0, 1, m)
	y = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2) + err
	return train_test_split(X, y, test_size=2 / 3)


def k_cross(X, y, k):
	err_per_rank = np.empty((15, 2))  # (test_err, train_err) for every rank
	model = LinearRegression()
	kf = KFold(n_splits=k)
	for rank in range(1, 16):
		err_specific_rank = np.empty((k, 2))  # (test_err, train_err) for every k
		poly = PolynomialFeatures(rank)
		idx = 0
		for train, test in kf.split(X, y):
			poly_train_X = poly.fit_transform(X[train])
			poly_test_X = poly.fit_transform(X[test])
			model.fit(poly_train_X, y_train)
			err_specific_rank[idx, 0] = mean_squared_error(y_test, model.predict(poly_test_X))
			err_specific_rank[idx, 1] = mean_squared_error(y_train,
			                                               model.predict(poly_train_X))
			idx += 1
		err_per_rank[rank - 1] = err_specific_rank.mean(axis=0)
		plt.plot(np.arange(1, 16), err_per_rank[:, 0], label='validation')
		plt.plot(np.arange(1, 16), err_per_rank[:, 1], label='train')
		plt.title('validation/train errors vs. polynomial rank')


if __name__ == '__main__':
	X_train, X_test, y_train, y_test = gen_data()
	k_cross(X_train, y_train, k=5)
	plt.show()