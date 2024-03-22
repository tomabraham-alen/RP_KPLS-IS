# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:48:35 2018
"""
import numpy as np
import scipy as sci
from scipy.stats import norm
from scipy.stats import multivariate_normal
import openturns as ot
from minimax_tilting_sampler import TruncatedMVN
from gaussian_process import GaussianProcess
from scipy import stats
import time
from openturns.usecases import cantilever_beam
import matplotlib.pyplot as plt
from smt.surrogate_models import KPLS


def U_criterion(y, MSE):
    return np.abs(y) / np.sqrt(MSE)


def EFF_criterion(y, MSE):
    # search of the best learning point x_star among the MC population using the EFF criterion
    epsilon = 2.0 * np.sqrt(MSE)
    X = y / np.sqrt(MSE)
    X1 = (-epsilon - y) / np.sqrt(MSE)
    X2 = (epsilon - y) / np.sqrt(MSE)
    X_cdf = sci.stats.norm.cdf(-X)
    X_pdf = sci.stats.norm.pdf(-X)
    X1_cdf = sci.stats.norm.cdf(X1)
    X1_pdf = sci.stats.norm.pdf(X1)
    X2_cdf = sci.stats.norm.cdf(X2)
    X2_pdf = sci.stats.norm.pdf(X2)
    EFF = y * (2.0 * X_cdf - X1_cdf - X2_cdf) - np.sqrt(MSE) * (2.0 * X_pdf - X1_pdf - X2_pdf) + epsilon * (
                X2_cdf - X1_cdf)
    return EFF


def exploration(y, MSE, cv_Pf, stop='U', cov_max=0.05):
    need_learning = False
    need_enriching = False
    if stop == "EFF":
        improve_criterion = EFF_criterion(y, MSE)
        need_learning = improve_criterion.max() > 1e-3
        need_enriching = cv_Pf > cov_max
    elif stop == "U":
        improve_criterion = -U_criterion(y, MSE)
        need_learning = improve_criterion.max() > -2
        need_enriching = cv_Pf > cov_max
    return improve_criterion, need_learning, need_enriching


def calculate_importance_weight(x, original_dist, importance_dist):
    """
    Calculate the importance weight for a given point based on two multivariate normal distributions.

    Parameters:
    - x: The point at which to evaluate the importance weight.
    - initial_mean: The mean vector of the initial distribution.
    - initial_covariance: The covariance matrix of the initial distribution.
    - importance_mean: The mean vector of the importance distribution.
    - importance_covariance: The covariance matrix of the importance distribution.

    Returns:
    - importance_weight: The importance weight at the given point.
    """
    initial_pdf = original_dist.computePDF(x)
    importance_pdf = importance_dist.computePDF(x)

    # Avoid division by zero
    if importance_pdf == 0:
        return 0.0

    importance_weight = initial_pdf / importance_pdf
    return importance_weight


def Vb_AGP(mu1, mu2, cov, performance_function, initial_distribution, distribution_chosen, inputSample,
           initial_IS_sample_size, initial_doe_size=None, learning_function='U', cov_max=0.1, hot_start=False, path='',
           corr='matern52'):
    var_G_E_X_l = list()
    var_X_E_G_l = list()
    var_tot_l = list()
    var_tot = 0.
    var_tot_inf = 0.
    var_tot_sup = 0.

    if hot_start == False:  # new initial DoE, new initial MC
        # initial MC population
        IS_sample_size = int(initial_IS_sample_size)
        ImportaneSamples = np.array(distribution_chosen.getSample(IS_sample_size))
        # ImportaneSamples = ImportaneSamples.transpose()
        # print(ImportaneSamples)
        # initial DoE generation
        lhs = ot.MonteCarloExperiment(initial_distribution, initial_doe_size)
        # sample, weight = lhs.generateWithWeights()

        DoE = np.array(lhs.generate())
        # DoE = np.array(inputSample)
        # DoE_size = DoE.shape[0]
        # inital DoE evaluation
        print("--------------------------------------")
        print("Evalulation of the perfomance function on the initial doe of size ", initial_doe_size)
        print("--------------------------------------")
        # Evaluation of the performance function - deflection of cantilever beam
        Y = np.zeros((initial_doe_size, 1))
        i = 0

        print(" All DoE values are:")
        for x in DoE:
            print(x)
        for x in DoE:
            Y[i, 0] = performance_function(x)
            print(Y[i, 0])
            i = i + 1
        print("--------------------------------------")
        print("Saving the initial DoE ")
        print("--------------------------------------")
        ImportaneSamples.dump(path + str('ImportaneSamples_init.pk'))
        DoE.dump(path + str('DoE_init.pk'))
        Y.dump(path + str('Y_init.pk'))

    elif hot_start == 'init':  # reuse of the previous initial MC population and inital DoE
        ImportaneSamples = np.load(path + str('ImportaneSamples_init.pk'), allow_pickle=True)
        IS_sample_size = ImportaneSamples.shape[0]
        initial_IS_sample_size = IS_sample_size
        DoE = np.load(path + str('DoE_init.pk'), allow_pickle=True)
        DoE_size = DoE.shape[0]
        Y = np.load(path + str('Y_init.pk'), allow_pickle=True)
        print("--------------------------------------")
        print("Reading the initial DoE of size ", DoE_size)
        print("--------------------------------------")


    elif hot_start == 'init_doe':  # new MC populaton and reuse of the initial DoE
        IS_sample_size = initial_IS_sample_size
        ImportaneSamples = np.array(distribution_chosen.getSample(IS_sample_size))
        initial_IS_sample_size = IS_sample_size
        DoE = np.load(path + str('DoE_init.pk'), allow_pickle=True)
        DoE_size = DoE.shape[0]
        Y = np.zeros((DoE_size, 1))
        i = 0
        for x in DoE:
            Y[i, 0] = performance_function(x)
            print(Y[i, 0])
            i = i + 1
        print("--------------------------------------")
        print("Reading the initial DoE of size ", DoE_size)
        print("--------------------------------------")



    elif hot_start == True:  # reuse of the MC population and DoE (last update)
        ImportaneSamples = np.load(path + str('ImportaneSamples.pk'), allow_pickle=True)
        IS_sample_size = ImportaneSamples.shape[0]
        initial_IS_sample_size = IS_sample_size
        DoE = np.load(path + str('DoE.pk'), allow_pickle=True)
        DoE_size = DoE.shape[0]
        Y = np.load(path + str('Y.pk'), allow_pickle=True)
        print("--------------------------------------")
        print("Hot start with a DoE of size ", DoE_size)
        print("--------------------------------------")

    # construction of the GP surrogate
    stochastic_dim = DoE.shape[1]
    theta0 = np.array([0.1] * stochastic_dim)
    thetaL = np.array([1e-6] * stochastic_dim)
    thetaU = np.array([100.0] * stochastic_dim)
    gp_g = GaussianProcess(corr=corr, theta0=theta0, thetaL=thetaL, thetaU=thetaU)
    # Vb-AGP loop
    n_iter = 0
    convergence = False
    n_iter = 1
    Y_train = Y.copy()
    X_train = DoE.copy()
    need_learning = True
    need_enriching = True
    ImportaneSamples_candidates = ImportaneSamples.copy()

    '''KRIGING METAMODEL PART'''
    gp_g.fit(X_train, Y_train)

    '''smt_KPLS MODEL PART'''
    sm = KPLS(theta0=[1e-2])
    sm.set_training_values(X_train, Y_train)
    sm.train()
    ncomp = sm.options["n_comp"]
    print("\n The model automatically choose " + str(ncomp) + " components.")

    cv_max = 0.02
    list_res = list()
    # In[15]:
    iter_max = 1000
    i = 0
    Pf = []
    cv_Pf = []

    while n_iter < iter_max and (need_learning or need_enriching):
        print("______________________________________________________________________________")
        print("iter =", n_iter)
        # Global prediction
        t1 = time.time()
        y_glob_kpls = sm.predict_values(ImportaneSamples)
        KPLS_variance = sm.predict_variances(ImportaneSamples)

        y_glob, MSE_glob = gp_g.predict(ImportaneSamples, eval_MSE=True)
        y_pred, MSE_pred = gp_g.predict(ImportaneSamples_candidates, eval_MSE=True)
        indicatrice_glob = np.zeros(np.size(y_glob))
        print(" length of y_glob_KRIGING : \t" + str(np.size(y_glob)))
        print(" length of y_glob_KPLS : \t" + str(np.size(y_glob_kpls)))

        print("Importance Samples size:", np.size(ImportaneSamples))

        # Weight calculation part
        weights = []
        norm_imp_weights = []
        for n in range(0, np.size(y_glob)):
            # print("Weight iteration",n)
            # print(ImportaneSamples[n])
            weights.append(calculate_importance_weight(ImportaneSamples[n], initial_distribution, distribution_chosen))
        # print(weights)
        for j in range(0, np.size(y_glob)):
            if y_glob_kpls[j] > 0.21:
                indicatrice_glob[j] = 1 * weights[j]
        # indicatrice_glob = (1-np.sign(y_glob))/2
        Pf.append(np.sum(indicatrice_glob) / ImportaneSamples.shape[0])
        print("Pf=", Pf)
        if Pf[i] > 0.0:
            cv_Pf.append(np.sqrt(np.var(indicatrice_glob) / ImportaneSamples.shape[0] / Pf[i]))
            t2 = time.time()
            print("Pf = " + str(Pf[i]) + " CV = " + str(cv_Pf[i]) + " N_MC = " + str(
                ImportaneSamples.shape[0]) + " DOE = " + str(X_train.shape[0]) + " Time: " + str(t2 - t1))
            improve_criterion, need_learning, need_enriching = exploration(y_glob_kpls, KPLS_variance, cv_Pf[i],
                                                                           stop=learning_function, cov_max=cv_max)
            print(np.size(improve_criterion))
            print("Convergence max = " + str(improve_criterion.max()))
            if need_learning:  # Enriching DOE
                which_x_to_add = np.argmax(improve_criterion)
                val_star = improve_criterion[which_x_to_add]
                x_star = ImportaneSamples_candidates[which_x_to_add, :]
                ImportaneSamples_candidates = np.delete(ImportaneSamples_candidates, which_x_to_add, 0)
                y_star = performance_function(np.atleast_2d(x_star))
                print("Enriching DOE: y star = ", y_star)
                X_train = np.vstack([X_train, np.atleast_2d(x_star)])
                Y_train = np.hstack([Y_train, y_star])
                sm.set_training_values(X_train, Y_train)
                sm.train()
            elif need_enriching:  # Enriching MC
                new_MC = np.array(distribution_chosen.getSample(IS_sample_size))
                # new_MC = new_MC.transpose()
                ImportaneSamples_candidates = np.vstack([ImportaneSamples_candidates, new_MC])
                ImportaneSamples = np.vstack([ImportaneSamples, new_MC])
                print("Enriching MC")
            else:
                print("Done")
        n_iter += 1
        i += 1
    list_res.append(np.array([Pf, var_X_E_G_l, var_G_E_X_l, improve_criterion.max()],dtype=object))
    # print(Pf)
    # print(cv_Pf)
    return Pf, X_train, n_iter, gp_g, Y_train, ImportaneSamples.shape[0], cv_Pf, sm


def cbeam(X):
    disp = (X[1] * X[2] ** 3) / (3 * X[0] * X[3])
    return disp


##### Distributions

stochastic_dim = 4

path = ''

cb = cantilever_beam.CantileverBeam()
distribution = cb.distribution
# print(type(distribution))
model = cb.model
E_distribution = ot.Beta(0.9, 3.5, 65.0e9, 75.0e9)
F_distribution = ot.LogNormalMuSigma()([300.0, 30.0, 0.0])
L_distribution = ot.Uniform(2.5, 2.6)
I_distribution = ot.Beta(2.5, 4.0, 1.3e-7, 1.7e-7)

fct_test = cbeam
distrib_test = distribution
initial_doe_size = 16
initial_IS_sample_size = 100
cov_max = 0.05

# IS additions

# Threshold event creation
vect = ot.RandomVector(distribution)
G = ot.CompositeRandomVector(model, vect)
event = ot.ThresholdEvent(G, ot.Greater(), 0.21)
# FORM configuration
optimAlgo = ot.Cobyla()
optimAlgo.setMaximumEvaluationNumber(300)
optimAlgo.setMaximumAbsoluteError(1.0e-10)
optimAlgo.setMaximumRelativeError(1.0e-10)
optimAlgo.setMaximumResidualError(1.0e-10)
optimAlgo.setMaximumConstraintError(1.0e-10)
# FORM analysis
algo = ot.FORM(optimAlgo, event, distribution.getMean())
algo.run()
result = algo.getResult()
# Printing results
standardSpaceDesignPoint = result.getStandardSpaceDesignPoint()
physicalSpaceDesignPoint = result.getPhysicalSpaceDesignPoint()
print(physicalSpaceDesignPoint)
# Standard Gaussian centered around SPDP-standardSpaceDesignPoint
dimension = distribution.getDimension()
mu = np.array(distribution.getMean())
cov = distribution.getCovariance()

importanceDistribution = ot.Normal(mu, cov)
importanceDistribution.setMean(physicalSpaceDesignPoint)

# InputSamples to build Initial DoE for Kriging meta model
result1 = result.getOptimizationResult()
inputSampleComplete = result1.getInputSample()
inputSampleTrunc = np.zeros((16, 4))
for i in range(0, 16, 1):
    for j in range(0, 4, 1):
        inputSampleTrunc[i][j] = inputSampleComplete[i][j]

distrib_test = importanceDistribution

# CDF (No scaling needed)
ub = np.ones_like(mu) * np.inf
cdf1 = distribution.computeCDF(ub)
cdf2 = importanceDistribution.computeCDF(ub)
print(cdf1)
print(cdf2)

Pf_ub = []
Pf_lb = []
var = []

start_time = time.time()

PF, DoE, n_iter, gp_g, Y, IS_sample_size, cov_tot, sm = Vb_AGP(mu, physicalSpaceDesignPoint, cov, fct_test, distribution,
                                                           distrib_test, inputSampleTrunc, initial_IS_sample_size,
                                                           initial_doe_size=initial_doe_size, path=path,
                                                           cov_max=cov_max, corr='matern52')
end_time = time.time()
total_time = end_time - start_time

print(n_iter)
size = len(PF)
print(size)
print("Pf=", PF)
print("COV=", cov_tot)
for i in range(0, size):
    var.append(((cov_tot[i] * PF[i])) ** 2)
    Pf_ub.append((PF[i] + 1.96 * np.sqrt(var[i])))
    Pf_lb.append((PF[i] - 1.96 * np.sqrt(var[i])))

num_iter = np.linspace(1, n_iter - 1, size)
print(num_iter)
print("Pf_lb=", Pf_lb)
print("Pf_ub=", Pf_ub)
print("Variance=", var)

print(f"total time taken is : {total_time}")

plt.title('Failure Probability Vs Number of iterations of KPLS-IS')
plt.xlabel('Number of iterations of KPLS-IS')
plt.ylabel('Failure Probability (Pf)')
# plt.ylim(0.025, 0.05)
plt.plot(num_iter, PF, label="Mean")
plt.plot(num_iter, Pf_ub, label="Upper Bound")
plt.plot(num_iter, Pf_lb, label="Lower Bound")
plt.legend(loc="best")
plt.show()




