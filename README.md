
# Demo confidence judgments

Small demonstration of the difficulty to make decision confidence judgments for learners which do not rely on distributional estimates. 

## The task
On one of many repeated trials of the experiment, the learner is supposed to infer the latent variable of a Bernoulli process. This is similar to multiple tosses of a loaded coin of which we would like to infer the tendency to land "head" or "tail" .  Uncertainty and thus difficulty of the task is increased when the sample size is small. 

The relevant statistics per trial are:
-- proportion of heads
-- sample size

Estimate:
-- whether the process (coin) favors "head" or "tail" (decision)  and
-- the confidence that your decision actually turns out to be correct

## The learners
We compare three different approaches to this problem by evaluating how their "deviation" from calibrated confidence judgments evolves over time.

1) Probabilistic agent knowing the generative model but having a slightly mistuned prior distribution
2) A model-free agent learning a direct mapping onto confidence judgments from the feedback about decision correctness. This mapping is conditional on sample size.
	1) Batch learning: Perfect memory of all relevant statistics.
	2) Iterative on-line learning only from the last decision (stochastic gradient descent) 

## Usage
Run the module `demo_comparison_confidence_judgments.py` as script or use it interactively through its `main()` method. The module `agents.py` implements the functionality of the learners and is imported by default.

## The result
`![deviation over trial index](comparison_agents.png)`
The probabilistic agent does not need feedback and makes approximately calibrated confidence judgments from the first trial. A mismatched prior distribution was introduced to show that its effects are not severe. In principle, the agent could use the feedback to learn this distribution over time by an hierarchical extension of its representation of the generative model.

The mapping-based agents, on the other hand, depend on feedback and they take many trials to get anywhere near the performance of the probabilistic agent even in this relatively simple problem. In addition, the assumption, or feasibility, of batch learning is a rather strong one because all data needs to be memorized. However, for iterative on-line learning the performance gap is larger still.

## Further informtion
This demonstration is based on:
**Probabilistic models for human judgments about uncertainty in intuitive inference tasks**, doctoral thesis, Philipp Schustek (Section 2.1.2).

