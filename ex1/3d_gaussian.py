import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr

mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T


def get_orthogonal_matrix(dim):
	H = np.random.randn(dim, dim)
	Q, R = qr(H)
	return Q


def plot_3d(x_y_z, title, fig_num):
	'''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
	fig = plt.figure(fig_num)
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
	ax.set_xlim(-5, 5)
	ax.set_ylim(-5, 5)
	ax.set_zlim(-5, 5)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.title.set_text(title)
	plt.savefig(f'plots/{title}.png')


def plot_2d(x_y, title, fig_num):
	'''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
	fig = plt.figure(fig_num)
	ax = fig.add_subplot(111)
	ax.scatter(x_y[0], x_y[1], s=1, marker='.')
	ax.set_xlim(-5, 5)
	ax.set_ylim(-5, 5)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.title.set_text(title)
	plt.savefig(f'plots/{title}.png')


def multivariative_gaussians():
	"""
	code for generating plots for this section of the exercise. calculates the tranformations,
	generates and saves the plots. if there are path saving problems disble the savefig lines I
	added in plot3D and plot2D
	:return:
	"""
	plot_3d(x_y_z=x_y_z, title='Identity covariance matrix', fig_num=1)
	S = np.diag([0.1, 0.5, 2])
	S_x_y_z = S @ x_y_z
	plot_3d(x_y_z=S_x_y_z, title='transformed data diagonal matrix', fig_num=2)
	print(f'covariance matrix after diagonal transformation:\n{np.cov(S_x_y_z)}')
	random_mat = get_orthogonal_matrix(dim=3)
	rand_x_y_z = random_mat @ x_y_z
	plot_3d(x_y_z=rand_x_y_z, title='transformed data random matrix', fig_num=3)
	print(f'covariance matrix after random orthogonal transformation:\n{np.cov(rand_x_y_z)}')
	plot_2d(x_y=x_y_z[:2], title='2D projection of gaussian', fig_num=4)
	limited_z = x_y_z[:2, np.logical_and(-0.4 < x_y_z[2], x_y_z[2] < 0.1)]
	plot_2d(x_y=limited_z, title='2D projection of gaussian where z in [-0.1,0.4]', fig_num=5)
	plt.show()


def concentration_inequalities():
	"""
	code for question 16. first is subsection a then b & c.
	:return:
	"""
	data = np.random.binomial(1, 0.25, (100000, 1000))
	epsilon = np.array([0.5, 0.25, .1, .01, .001])
	m = np.arange(1, data.shape[1] + 1)
	avgs = np.cumsum(data, axis=1) / m
	fig = plt.figure(6)
	ax = fig.add_subplot(111)
	ax.scatter(x=m, y=avgs[0, :], c="b", s=3)
	ax.scatter(x=m, y=avgs[1, :], c='r', s=3)
	ax.scatter(x=m, y=avgs[2, :], c="g", s=3)
	ax.scatter(x=m, y=avgs[3, :], c='c', s=3)
	ax.scatter(x=m, y=avgs[4, :], c="y", s=3)
	ax.grid()
	ax.set_xlabel('No. of tosses')
	ax.set_ylabel('ratio of successful tosses')
	ax.title.set_text('Successful tosses vs. No. of tosses, 5 different sets')
	plt.savefig('plots/16a.png')
	fig_num = 7
	var_x = 15 / 16
	for epsil in epsilon:
		fig = plt.figure(fig_num)
		ax = fig.add_subplot(111)
		chebyshev = np.clip(var_x / m / epsil ** 2, 0, 1)
		hoeffding = np.clip(2 * np.exp(-2 * m * epsil ** 2), 0, 1)  # 0<X<1 -> a=0,b=1 -> (b-a)^2=1
		percentages = np.sum(np.abs(avgs - 0.25) >= epsil, axis=0) / 100000
		ax.scatter(x=m, y=chebyshev, c='b', label='Chebyshev', s=5)
		ax.scatter(x=m, y=hoeffding, c='y', label='Hoeffding', s=5)
		ax.scatter(x=m, y=percentages, c='g', label='percentage', s=5)
		ax.grid()
		ax.legend()
		ax.set_xlabel('No. of tosses')
		ax.set_ylabel('Bound')
		ax.title.set_text(f'Bounds for epsilon = {epsil}')
		plt.savefig(f'plots/16b_epsilon_{epsil}.png')
		fig_num += 1
	plt.show()


if __name__ == '__main__':
	multivariative_gaussians()
	#concentration_inequalities()

