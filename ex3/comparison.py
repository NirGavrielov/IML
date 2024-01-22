import numpy as np, matplotlib.pyplot as plt, models

figure_num = 1

m_vals = np.array([5, 10, 15, 25, 70])
f = lambda x: np.sign(x @ [0.3, -0.5] + 0.1)
lda = models.LDA()
percep = models.Perceptron()
svm = models.SVM()


def draw_points(m):
	points = np.random.multivariate_normal([0, 0], np.eye(2), m)
	return points, f(points)


def plotting():
	global figure_num
	for m in m_vals:
		X, y = draw_points(m)
		positive = y == 1
		plt.figure(figure_num)
		plt.scatter(X[positive, 0], X[positive, 1], c='b', label='1')
		plt.scatter(X[~positive, 0], X[~positive, 1], c='orange', label='-1')
		percep.fit(X, y), svm.fit(X, y)
		f_true = lambda x: (0.3 * x + 0.1) / 0.5
		f_perc = lambda x: -(percep.model[1] * x + percep.model[0]) / percep.model[2]
		f_svm = lambda x: -(svm.model.coef_[0][0] * x + svm.model.intercept_[0]) / \
		                  svm.model.coef_[0][1]
		x = np.array([X[:, 0].min(), X[:, 0].max()])
		plt.plot(x, f_true(x), label='True')
		plt.plot(x, f_perc(x), label='Perceptron')
		plt.plot(x, f_svm(x), label='SVM')
		plt.xlabel('first feature')
		plt.ylabel('second feature')
		plt.title(f'results for #samples = {m}')
		plt.legend()
		# plt.savefig(f'plots\q9, #samples={m}.png')
		figure_num += 1


def comparison():
	global figure_num
	k, repeat = 10000, 500
	mean_accuracies = np.empty((m_vals.size, 3))  # line is #iteration, cols are lda, svm, perceptron
	for j, m in enumerate(m_vals):
		accuracies = np.empty((repeat, 3))  # line is #iteration, cols are lda, svm, perceptron
		for i in range(repeat):
			X, y = draw_points(m)
			while -1 not in y or 1 not in y:  # have to assure both values in array
				X, y = draw_points(m)
			lda.fit(X, y), svm.fit(X, y), percep.fit(X, y)
			t_X, t_y = draw_points(k)
			accuracies[i, 0] = lda.score(t_X, t_y)['accuracy']
			accuracies[i, 1] = svm.score(t_X, t_y)['accuracy']
			accuracies[i, 2] = percep.score(t_X, t_y)['accuracy']
			mean_accuracies[j, :] = accuracies.mean(axis=0)
	plt.scatter(m_vals, mean_accuracies[:, 0], label='LDA')
	plt.scatter(m_vals, mean_accuracies[:, 1], label='SVM')
	plt.scatter(m_vals, mean_accuracies[:, 2], label='Perceptron')
	plt.xticks(m_vals)
	plt.legend()
	plt.xlabel('# samples'), plt.ylabel('accuracy'), plt.title('accuracy of different models')
	# plt.savefig('plots\comparison.png')


if __name__ == '__main__':
	plotting()
	comparison()
	plt.show()
