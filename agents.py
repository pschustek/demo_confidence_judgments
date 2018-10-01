# -*- coding: utf-8 -*-
"""
Implementation of decision making and confidence judgments
"""
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize


class Decision:

    def __init__(self, nh, n):
        self.nh = nh
        self.nt = n - nh
        self.n = n
        self.num_trials = len(nh)
        self.make_decision()
        self.decision_correct = None

    def make_decision(self):
        q = np.divide(self.nh, self.n)
        decision = np.zeros(self.num_trials)
        decision[q > 0.5] = 1
        # Break ties at random
        ties = q == 0.5
        decision[ties] = np.random.binomial(1, 0.5, np.sum(ties))
        self.decision = decision

    def decision_feedback(self, real_majority):
        self.decision_correct = np.zeros(self.num_trials)
        self.decision_correct[real_majority == self.decision] = 1

    def align_to_decision(self, x):
        return 2 * (self.decision - 0.5) * (x - 0.5) + 0.5


class ProbabilisticConfidence(Decision):

    def __init__(self, nh, n, h0=4, t0=4):
        Decision.__init__(self, nh, n)
        self.h0 = h0
        self.t0 = t0

        # Confidence in a majority of 'heads'
        conf_h = 1 - stats.beta.cdf(0.5, self.nh + self.h0, self.nt + self.t0)
        self.confidence_h = conf_h
        self.confidence_d = self.align_to_decision(conf_h)


class MappingBased(Decision):

    def __init__(self, nh, n):
        Decision.__init__(self, nh, n)
        self.set_n = np.unique(n)
        self.W = None

    # Define parameterized stimulus-response mapping
    def decision_confidence_map(dq, w):
        return 1/(1+np.exp(-(w[0]*np.abs(dq-0.5)+w[1]*np.abs(dq-0.5)**3)))

    def random_initial_weights(self, mu, sigma):
        # Array for weights: (n,w)
        self.W = np.random.normal(mu, sigma, (self.set_n.size, 2))

    def confidence_batch_learning(self):
        # Decision-aligned proportion of 'heads'
        dq = self.align_to_decision(np.divide(self.nh, self.n))

        # Initialize memory for tracking of the number of correct responses (C),
        # the number of responses (N) and their quotient (Q) for each condition (dq, N)
        Q, C, N = [], [], []
        for j, n in enumerate(self.set_n):
            # Possible decision-aligned proportion of 'heads' for each sample size
            Q.append(np.arange(np.ceil(n/2), n+1) / n)
            # Initialize to C/N = 0.5
            C.append(np.ones(Q[j].size)*0.5)
            N.append(np.ones(Q[j].size, dtype=int))

        # Index (idn) for sample size n for all trials
        M = self.n[:, np.newaxis] == self.set_n
        n_indices = np.tile(np.arange(*self.set_n.shape), (self.num_trials, 1))[M]

        # Cycle through trials
        conf_batch = np.zeros(self.num_trials)
        for t in range(self.num_trials):

            # Select corresponding weights w by using boolean mask for sample size
            idn = n_indices[t]

            # Report decision confidence
            conf_batch[t] = MappingBased.decision_confidence_map(dq[t], self.W[idn, :])

            # Memorize outcomes
            idq = Q[idn] == dq[t]
            if self.decision_correct[t]:
                C[idn][idq] += 1

            N[idn][idq] += 1

            # Train function
            X = Q[idn]
            Y = np.divide(C[idn], N[idn])

            def ferr(w):
                return np.sum((MappingBased.decision_confidence_map(X, w) - Y) ** 2)

            res = minimize(ferr, self.W[idn, :], method='BFGS', jac=None,
                           hess=None, options={'gtol': 1e-8, 'disp': False})
            # Update
            self.W[idn, :] = res.x

        self.confidence_d = conf_batch

    def confidence_sgd_learning(self, step_size):
        # On-line learning through stochastic gradient descent
        # Array for weights; (N,w)

        # Decision-aligned proportion of 'heads'
        dq = self.align_to_decision(np.divide(self.nh, self.n))

        # Index (idn) for sample size n for all trials
        M = self.n[:, np.newaxis] == self.set_n
        n_indices = np.tile(np.arange(*self.set_n.shape), (self.num_trials, 1))[M]

        # Cycle through trials
        conf_sgd = np.zeros(self.num_trials)
        for t in range(self.num_trials):

            idn = n_indices[t]
            w = self.W[idn, :]

            # Report decision confidence (repeatedly used)
            conf = MappingBased.decision_confidence_map(dq[t], w)
            conf_sgd[t] = conf

            # Gradient
            derivative_base = 2*(conf-self.decision_correct[t])*conf*(1-conf)
            grad = np.array([derivative_base*abs(dq[t]-0.5),
                             derivative_base*abs(dq[t]-0.5)**3])

            # Adjust weights
            self.W[idn, :] = self.W[idn, :] - step_size*grad

        self.confidence_d = conf_sgd
