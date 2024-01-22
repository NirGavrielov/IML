import numpy as np, matplotlib.pyplot as plt, time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# train_data = np.loadtxt("mnist_train.csv", delimiter=",")
# test_data = np.loadtxt("mnist_test.csv", delimiter=",")
# np.save('train', train_data), np.save('test', test_data)

train_data, test_data = np.load('mnist_train.npy'), np.load('mnist_test.npy')

x_train, y_train = train_data[:, 1:], train_data[:, 0]
x_test, y_test = test_data[:, 1:], test_data[:, 0]

train_images = np.logical_or((y_train == 0), y_train == 1)
test_images = np.logical_or((y_test == 0), y_test == 1)
x_train, y_train = x_train[train_images], y_train[train_images]
x_test, y_test = x_test[test_images], y_test[test_images]


def rearrange_data(X):
	n = X.shape[0]
	return X.reshape((n, 784))


def draw_points(m):
	while True:
		inds = np.random.randint(0, y_train.size, m)
		points, results = x_train[inds], y_train[inds]
		if 0 in results and 1 in results:
			return points, results


def comparison():
	m_values = np.array([50, 100, 300, 500])
	svm = SVC(C=1e10, kernel='linear')
	knn = KNeighborsClassifier()
	tree = DecisionTreeClassifier(max_depth=12)
	logistic = LogisticRegression(solver='liblinear')
	classifiers = [svm, knn, tree, logistic]
	classifier_names = ['svm', 'knn', 'tree', 'logistic']
	means = np.empty((m_values.size, len(classifiers)))
	times = np.empty((m_values.size, len(classifiers)))
	repeat = 50
	for idx_m, m in enumerate(m_values):
		accuracies = np.empty((repeat, len(classifiers)))
		for idx_class, classifier in enumerate(classifiers):
			start = time.time()
			for i in range(repeat):
				X, y = draw_points(m)
				classifier.fit(X, y)
				y_hat = classifier.predict(x_test)
				accuracies[i, idx_class] = (y_test == y_hat).sum() / y_test.size
			times[idx_m, idx_class] = time.time() - start
		means[idx_m] = np.mean(accuracies, axis=0)
	for i in range(len(classifiers)):
		plt.scatter(m_values, means[:, i], label=classifier_names[i])
	plt.xticks(m_values)
	plt.xlabel('# samples')
	plt.ylabel('accuracy')
	plt.title('Various classifiers accuracy on mnist data')
	plt.legend()
	# plt.savefig('plots\mnist.png')
	plt.show()
	print(f'measured times of 50 repetitions in ms, '
	      f'rows are m = {m_values}\t cols are {classifier_names}\n', times * 1000)


if __name__ == '__main__':
	comparison()
