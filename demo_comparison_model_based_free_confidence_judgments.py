# -*- coding: utf-8 -*-
"""
Demo of confidence judgments of two different learners

More text
"""
import numpy as np
import matplotlib.pyplot as plt
import agents as ag
import sys
import pdb


def setup_task(beta_prior, num_trials):

    # Set of sample sizes
    set_n = np.arange(6,13)

    # Draws from multinomial distribution
    mask = np.random.multinomial(1, [1/set_n.size]*set_n.size, num_trials) == 1

    # Sample size for each trial
    num = np.tile(set_n, (num_trials,1))[mask]

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
#    pdb.set_trace()

    # For reproducible results
    np.random.seed(seed)

    # Set up experiment
    nh, num, real_majority, ph = setup_task(beta_prior, num_trials)

    ideal = ag.ProbabilisticConfidence(nh, num, beta_prior, beta_prior)
    dec = ideal.decision

    # Confidence of ideal observer estimates the probability of making correct choices
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
    mapping_sgd.random_initial_weights(0, 2) # TODO W0
    mapping_sgd.confidence_sgd_learning(step)

    def movmean(x, N):  # TODO handle edge effects
        return np.convolve(x, np.ones(N)/N)[(N-1):]

    devA2Batch = np.abs(mapping_batch.confidence_d - prob_correct_choice)
    devA2SGD = np.abs(mapping_sgd.confidence_d - prob_correct_choice)
    devA1 = np.abs(prob_mismatched.confidence_d - prob_correct_choice)

    num_trials = len(nh)    # TODO improve
    leg = plt.plot(np.arange(num_trials)+1, movmean(devA1,30),
             np.arange(num_trials)+1, movmean(devA2Batch,30),
             np.arange(num_trials)+1, movmean(devA2SGD,30))

    font = {'family': 'serif', 'size': 12}
    plt.xlabel('trial number', font)
    plt.ylabel('deviation', font)
    plt.legend(leg, ('A1', 'A2 batch', 'A2 sgd'), frameon=False)


if __name__ == '__main__':
    pdb.set_trace()
    main(*sys.argv[1:])