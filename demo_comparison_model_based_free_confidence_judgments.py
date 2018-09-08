# Demo of confidence judgments of two different learners

"""Demo of confidence judgments of two different learners

More text
"""
# Import modules
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#import pandas as pd

# For reproducible results
np.random.seed(262743)

# %% Ideal observer makes posterior-based confidence reports
def ideal_observer(nH, N, h=4, t=4):
    """Posterior-based confidence reports"""
    # Number of 'tails'
    nT = N - nH
    # Confidence in a majority of 'heads'
    return 1 - stats.beta.cdf(0.5, nH + h, nT + t)

# %% Parameter definitions
# To sample the 'coin bias'
betaParams = 4
numTrials = 2000

# Set of sample sizes
setN = np.arange(6,13)
# Draws from multinomial distribution
distN = np.random.multinomial(1, [1/setN.size]*setN.size, numTrials) == 1

# Sample size for each trial
num = np.tile(setN, (numTrials,1))[distN]

# Sample 'airplane proportions'
pH = np.random.beta(betaParams, betaParams, numTrials)

# What is the airplane majority
realMajority = pH > 0.5
# Break ties at random; just in case
ties = pH == 0.5
realMajority[ties] = np.random.binomial(1, 0.5, np.sum(ties))

# Sample the observations, number of 'heads'
nH = np.random.binomial(num, pH)
q = np.divide(nH,num)

# %% Simulate behavior
# Confidence of ideal observer
idealConfHeads = ideal_observer(nH, num)

# Decisions (the same for all agents)
decision = q > 0.5
ties = q == 0.5
decision[ties] = np.random.binomial(1, 0.5, np.sum(ties))
# Needed for learning from feedback below
decisionCorrect = realMajority == decision

# Decision-aligned proportion of 'heads'
dCq = 2*(decision-0.5)*(q-0.5) + 0.5

# Probability of making correct choices
dCIdeal = 2*(decision-0.5)*(idealConfHeads-0.5) + 0.5

# %% Agent 1 (Probabilistic)
# but with mismatched prior distribution
A1ConfHeads = ideal_observer(nH, num, 1, 1)

# Confidence estimate of Agent 1
dCA1 = 2*(decision-0.5)*(A1ConfHeads-0.5) + 0.5 

# %% Simulate behavior Agent 2: Batch learning
# Define parameterized stimulus-response mapping
def decision_confidence_A2(dq, w):
    return 1/(1+np.exp(-(w[0]*np.abs(dq-0.5)+w[1]*np.abs(dq-0.5)**3)))  

# Array for weights; (N,w)
A2_Wbatch = np.random.normal(0, 2, (setN.size,2))
# Use same initialization for on-line learning
A2_Wsgd = A2_Wbatch.copy()

# Initialize memory for tracking of the number of correct responses 
# and the number of responses for each condition (dq, N)
Q = []
C = []
N = []
for j, n in enumerate(setN):
    # Possible decision-aligned proportion of 'heads' for each sample size
    Q.append(np.arange(np.ceil(n/2),n)/n)
    # Initialize to C/N = 0.5
    C.append(np.ones(Q[j].size)*0.5)
    N.append(np.ones(Q[j].size, dtype=int))

# Cycle through trials
A2ConfBatch = np.zeros(numTrials)
for t in range(numTrials):
    # Index ths trials' sample size in setN
    idxN = [j for j,n in enumerate(setN) if n==num[t]][0]   # pick first element (only one)
    # Select corresponding weights w by using boolean mask for sample size
    maskN = num[0] == setN
    w = A2_Wbatch[maskN,:].squeeze()
    # Report decision confidence
    A2ConfBatch[t] = decision_confidence_A2(dCq[t], w)
    
    # Memorize outcomes
    idxq = Q[idxN] == dCq[t]
    if decisionCorrect[t]:
        C[idxN][idxq] = C[idxN][idxq] + 1
    N[idxN][idxq] = N[idxN][idxq] + 1
    
    # Train function
    X = Q[idxN]
    Y = np.divide(C[idxN], N[idxN])
    def ferr(w):
        return np.sum((decision_confidence_A2(X, w) - Y) ** 2)
    
    res = minimize(ferr, w, method='BFGS',
               jac=None, hess=None,
               options={'gtol': 1e-8, 'disp': False})
    A2_Wbatch[idxN,:] = res.x
    
# %% Simulate behavior Agent 2: On-line learning through stochastic gradient descent
# Same parameterized stimulus-response mapping as above
# Array for weights; (N,w)
# A2_Wsgd is initialized above
    
# Cycle through trials
A2ConfSGD = np.zeros(numTrials)
for t in range(numTrials):
    # Index ths trials' sample size in setN
    idxN = [j for j, n in enumerate(setN) if n==num[t]][0]   # pick first element (only one)
    # Select corresponding weights w by using boolean mask for sample size
    maskN = num[0] == setN
    w = A2_Wsgd[maskN,:].squeeze()
    # Report decision confidence
    A2ConfSGD[t] = decision_confidence_A2(dCq[t], w)
    C = decision_confidence_A2  # just a short-cut
    
    # SGD
    stepSize = 2
    grad = np.array([2*(C(dCq[t],w)-decisionCorrect[t])*C(dCq[t],w)*(1-C(dCq[t],w))*abs(dCq[t]-0.5),
        2*(C(dCq[t],w)-decisionCorrect[t])*C(dCq[t],w)*(1-C(dCq[t],w))*abs(dCq[t]-0.5)**3])
    A2_Wsgd[idxN,:] = A2_Wsgd[idxN,:] - stepSize*grad

#%%
def movmean(x, N):
    return np.convolve(x, np.ones(N)/N)[(N-1):]

devA2Batch = np.abs(A2ConfBatch-dCIdeal)
devA2SGD = np.abs(A2ConfSGD-dCIdeal)
devA1 = np.abs(dCA1-dCIdeal)

leg = plt.plot(np.arange(numTrials)+1, movmean(devA1,30),
         np.arange(numTrials)+1, movmean(devA2Batch,30), 
         np.arange(numTrials)+1, movmean(devA2SGD,30))

font = {'family': 'serif', 'size': 12}
plt.xlabel('trial number', font)
plt.ylabel('deviation', font)
plt.legend(leg, ('A1', 'A2 batch', 'A2 sgd'), frameon=False)
