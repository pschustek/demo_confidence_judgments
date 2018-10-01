# -*- coding: utf-8 -*-
"""
Demo confidence judgments

This is the main script which sets up the experiment, trains the learners
and plots the results
"""
import numpy as np
import matplotlib.pyplot as plt
import agents as ag
#import pdb


def setup_task(beta_prior, num_trials):
    """
    Set up the trials of the experiment and sample from the generative model

    Draw values for repeated experiment of the 'coin problem'

    Arguments:
        beta_prior:
            Parameter for symmetric Beta distribution
        num_trial:
            Number of trials of the whole experiment

    Returns:
        nh: array
            Number of 'head' in one trial
        num: array
            Sample size
        real_majority: array
            Actual (latent) majority of items in the urn
        ph: array
            Actual (latent) proportion of items in the urn
    """

    # Set of sample sizes
    set_n = np.arange(6, 13)

    # Draws from multinomial distribution
    mask = np.random.multinomial(1, [1/set_n.size]*set_n.size, num_trials) == 1

    # Sample size for each trial
    num = np.tile(set_n, (num_trials, 1))[mask]

    # Sample 'airplane proportions'
    ph = np.random.beta(beta_prior, beta_prior, num_trials)

    # What is the airplane majority
    real_majority = np.zeros(num_trials)
    real_majority[ph > 0.5] = 1
    # Break ties at random; just in case
    ties = ph == 0.5
    real_majority[ties] = np.random.binomial(1, 0.5, np.sum(ties))

    # Sample the observations, number of 'heads'
    nh = np.random.binomial(num, ph)

    return nh, num, real_majority, ph


def main(num_trials=600, seed=None, beta_prior=4, step=2):
    """
    Set parameters of the learners, compute the results and plot them.

    Arguments:
        num_trials:
            Number of trials of the whole experiment
        seed:
            Set seed for reproducible results
        beta_prior:
            Parameter for symmetric Beta-distribution modeling the
            base rates of the generative model
        step:
            Step-size for model-free learner using stochastic gradient descent

    Returns:
        None
    """

    # For reproducible results
    np.random.seed(seed)

    # Set up experiment
    nh, num, real_majority, ph = setup_task(beta_prior, num_trials)

    ideal = ag.ProbabilisticConfidence(nh, num, beta_prior, beta_prior)
    dec = ideal.decision

    # Confidence of ideal observer estimates the probability of
    # making correct choices
    prob_correct_choice = ideal.align_to_decision(ideal.confidence_h)

    # Decision rule is the same for all agents) except tie breaking
    # We enforce the same tie breaking by copying

    # Probabilistic agent with mismatching Beta-prior distribution
    prob_mismatched = ag.ProbabilisticConfidence(nh, num, 1, 1)
    prob_mismatched.decision = dec

    # Mapping based agent needs feedback to learn
    mapping_batch = ag.MappingBased(nh, num)
    mapping_batch.decision = dec
    # Learning from feedback
    mapping_batch.decision_feedback(real_majority)
    mapping_batch.random_initial_weights(0, 2)
    W0 = mapping_batch.W.copy()
    mapping_batch.confidence_batch_learning()

    # Mapping based agent needs feedback to learn
    mapping_sgd = ag.MappingBased(nh, num)
    mapping_sgd.decision = dec
    # Learning from feedback
    mapping_sgd.decision_feedback(real_majority)
    # Use same initial weights
    mapping_sgd.W = W0
    mapping_sgd.confidence_sgd_learning(step)

    def movmean(x, N):  # TODO handle edge effects
        """Moving mean over window of N trials

        Arguments:
            x: array
                Data to be smoothed
            N:
                Size of smoothing window

        Returns:
            Smoothed array of input x
        """
        return np.convolve(x, np.ones(N)/N)[(N-1):]

    devA2Batch = np.abs(mapping_batch.confidence_d - prob_correct_choice)
    devA2SGD = np.abs(mapping_sgd.confidence_d - prob_correct_choice)
    devA1 = np.abs(prob_mismatched.confidence_d - prob_correct_choice)

    num_trials = len(nh)
    leg = plt.plot(np.arange(num_trials) + 1, movmean(devA1, 30),
                   np.arange(num_trials) + 1, movmean(devA2Batch, 30),
                   np.arange(num_trials) + 1, movmean(devA2SGD, 30))

    font = {'family': 'serif', 'size': 12}
    plt.xlabel('trial number', font)
    plt.ylabel('deviation', font)
    plt.legend(leg, ('prob', 'map batch', 'map sgd'), frameon=False)
    plt.savefig('comparison_agents.png')


if __name__ == '__main__':
    main()