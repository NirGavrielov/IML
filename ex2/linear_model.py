import numpy as np, matplotlib.pyplot as plt, pandas as pd

figure_num = 1  # global variable used to organize the figures


def fit_linear_regression(X, y):
	"""

	:param X: a m*d numpy matrix, its rows are feature vectors
	:param y: response vector, m rows of labels
	:return: 1. w: coefficients vector
			 2. sing: array of the singular values of X
	"""
	u, s, v = np.linalg.svd(X)
	sing = np.diag(s)[np.nonzero(np.diag(s))]
	p_inv = np.linalg.pinv(X)
	w = p_inv @ y
	return w, sing


def predict(X, w):
	"""
	:param X: design matrix, m*d numpy matrix
	:param w: coefficients vector
	:return: y_hat: prediction of the trained model
	"""
	return X @ w


def mse(y, y_hat):
	"""

	:param y: response vector. m rows numpy array
	:param y_hat: prediction vector
	:return: MSE over samples
	"""
	return np.sum((y_hat - y) ** 2) / y.size


def load_data(filename):
	"""
	:param filename: path to csv file containing the data for the design matrix
	:return: the tuple: (X - design matrix, y - response vector)
	"""
	df = pd.read_csv(filename).dropna()
	y = np.abs(np.array(df['price']))
	X = np.abs(np.hstack((np.array(df.iloc[:, 3:16]), np.array(df.iloc[:, 19:21]))))
	return X, y


def plot_singular_values(sing):
	"""
	:param sing: collection of singular values
	plots them in descending order
	:return:
	"""
	global figure_num
	plt.figure(figure_num)
	figure_num += 1
	plt.plot(np.arange(1, sing.size + 1), sing, 'ro-', linewidth=2)
	plt.xticks(np.arange(sing.size + 1))
	plt.xlabel('Principal component')
	plt.ylabel('Singular value')
	plt.title('Design Matrix singular values')


def feature_evaluation(X, y):
	"""
	:param X: design matrix, the way we parsed the data non categorical features are in columns
	but 8-11
	:param y: response vector
	"""
	global figure_num
	features = ['bedrooms',	'bathrooms', 'sqft_living',	'sqft_lot', 'floors',
	            'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
	            'yr_renovated', 'sqft_living15', 'sqft_lot15']
	X_1 = np.hstack((X[:, :5], X[:, 6:]))
	stdev_x = np.std(X_1, axis=0)
	stdev_y = np.std(y)
	for i in range(X_1.shape[1]):
		feature = X_1[:, i]
		plt.figure(figure_num)
		figure_num += 1
		plt.scatter(feature, y)
		cov = np.mean(feature * y) - np.mean(feature) * np.mean(y)
		pears = cov / stdev_x[i] / stdev_y
		plt.title(f'response vs. {features[i]}, p. correlation = {pears:.2f}')
		plt.xlabel(f'{features[i]}')
		plt.ylabel('response')
		"""if i == X_1.shape[1] - 1:
			plt.savefig('lot_15.png')
		if i == 2:
			plt.savefig('living.png')"""


def MSE_check(X, y):
	"""
	creates a plot of the MSE as a function of percentage of data used
	:param X:
	:param y:
	:return:
	"""
	choice = np.random.choice(range(y.size), size=(int(y.size / 4),), replace=False)
	test = np.zeros(y.size, dtype=bool)
	test[choice] = True
	train = ~test
	X_train, y_train = X[train], y[train]
	X_test, y_test = X[test], y[test]
	size = 100
	mse_results = np.empty((size,))
	for i in range(size):
		max_ind = int(y.size * i / 100)  # i% percentage of data
		w1, sing1 = fit_linear_regression(X_train[:max_ind], y_train[:max_ind])
		mse_results[i] = mse(y_test, predict(X_test, w1))
	plt.figure(figure_num)
	plt.plot(np.arange(1, size + 1), mse_results)
	plt.xlabel('p% of the training set')
	plt.ylabel('MSE')
	plt.title('MSE over the test set as a function of p%')
	#plt.savefig('mse.png')


if __name__ == "__main__":
	figure_num = 1
	X, y = load_data('kc_house_data.csv')
	w, sing = fit_linear_regression(X, y)
	plot_singular_values(sing)
	MSE_check(X, y)
	feature_evaluation(X, y)
	plt.show()


