from openturns.usecases import stressed_beam
import openturns as ot
from openturns.usecases import cantilever_beam
import numpy as np
from create_symbolic_function import symboli_func

ot.Log.Show(ot.Log.NONE)

stochastic_dim = 40  # total number of dimensions of the problem
m = stochastic_dim
sig = 0.2  # std deviation of the distribution
temp_xvalues = np.ones(m)  # has no significance for the problem directly


"""Distribution and symbolic model creation"""
X_dist = ot.LogNormal()
X_dist.setParameter(ot.LogNormalMuSigma()([1, 0.2, 0.0]))
composed_dist = ot.ComposedDistribution([X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist,
                                         X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist,
                                         X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist,
                                         X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist]
                                        )
distribution = composed_dist
model = symboli_func(m, sig, temp_xvalues)

"""Event creation"""
vect = ot.RandomVector(distribution)
G = ot.CompositeRandomVector(model, vect)
event = ot.ThresholdEvent(G, ot.Less(), 1)


"""Setting up the algorithm for computing the probability of failure"""
experiment = ot.MonteCarloExperiment()
algo = ot.ProbabilitySimulationAlgorithm(event, experiment)
algo.setMaximumCoefficientOfVariation(0.05)
algo.setMaximumOuterSampling(int(1e4))
algo.run()


"""Accessing the result and computing the Pf"""
result = algo.getResult()
probability = result.getProbabilityEstimate()
print("Pf=", probability)