"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from ex4_tools import *

noise_ratio = 0

class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        global noise_ratio
        m = y.size
        D = np.ones((m, )) / m  # uniform distribution
        for i in range (self.T):
            self.h[i] = self.WL(D, X, y)
            y_hat = self.h[i].predict(X)
            epsil = (D * (y != y_hat).astype(np.int)).sum()
            self.w[i] = .5 * np.log(1/epsil - 1)
            D *= np.exp(- self.w[i] * y * self.h[i].predict(X))
            D /= D.sum()
        # drawing the distribution
        plt.figure(2)
        D1 = D / np.amax(D) * 10
        decision_boundaries(self, X, y, weights=D1, num_classifiers=self.T)
        plt.title('Looking into adaboost weights')
        # plt.savefig(f'plots/noise{int(noise_ratio * 100)}q16.png')

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        results = np.empty((X.shape[0], max_t))
        for i in range(max_t):
            results[:, i] = self.h[i].predict(X) * self.w[i]
        return np.sign(results.sum(axis=1))

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        @Nir: I understood as False_samples / N_samples
        """
        y_hat = self.predict(X, max_t)
        return (y != y_hat).sum() / y.size


if __name__=="__main__":
    T = 500
    train_samples = 5000
    test_samples = 200
    for noise_ratio in [0.01, 0.4]:
        trainX, trainy = generate_data(train_samples, noise_ratio=noise_ratio)
        testX, testy = generate_data(test_samples, noise_ratio=noise_ratio)
        ada = AdaBoost(DecisionStump, T)
        ada.train(trainX, trainy)
        q13errors = np.empty((T, 2))
        for t in range(T):
            q13errors[t] = ada.error(trainX, trainy, t), ada.error(testX, testy, t)
        plt.figure(1)
        plt.plot(np.arange(T), q13errors[:, 0], label='train error')
        plt.plot(np.arange(T), q13errors[:, 1], label='test error')
        plt.xlabel('number of trees')
        plt.ylabel('Error')
        plt.legend()
        plt.title('Adaboost Errors')
        # plt.savefig(f'plots/noise{int(noise_ratio * 100)}q13')
        t_maxes = np.array([5, 10, 50, 100, 200, 500])
        errors = np.empty(t_maxes.shape)
        fig = plt.figure(3)
        gs = GridSpec(2, int(t_maxes.size / 2))
        for idx, t_max in enumerate(t_maxes):
            decision_boundaries(ada, testX, testy, num_classifiers=t_max, fig=fig.add_subplot(gs[idx % 2, int(idx / 2)]))
            errors[idx] = ada.error(testX, testy, t_max)
        # plt.savefig(f'plots/noise{int(noise_ratio * 100)}q14')
        T_min, T_min_err = t_maxes[errors.argmin()], np.amin(errors)
        plt.figure(4)
        decision_boundaries(ada, trainX, trainy, num_classifiers=T_min)
        plt.title(f'test error minimizer T = {T_min} with training data, error={T_min_err}')
        # plt.savefig(f'plots/noise{int(noise_ratio * 100)}q15')
        plt.show()
