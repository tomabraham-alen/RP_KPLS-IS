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
from create_symbolic_function import symboli_func
from scipy import stats
import time
from openturns.usecases import cantilever_beam
import matplotlib.pyplot as plt
from smt.surrogate_models import KPLS

"""Learning criteria 1 to help find out the best value of input to add to DoE to do an enrichment"""


def U_criterion(y, MSE):
    return np.abs(y) / np.sqrt(MSE)


"""Learning criteria 1 to help find out the best value of input to add to DoE to do an enrichment"""


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


""" Returns the concerned learning criteria and checks if sample size enrichment is required or not"""


def exploration(sig, y, MSE, cv_Pf, stop='EFF', cov_max=0.05):
    need_learning = False
    need_enriching = False
    if stop == "EFF":
        improve_criterion = EFF_criterion(y, MSE)
        if np.average((MSE)) < (cov_max**2)/2:
            if (np.average((MSE)) + (sig**2)) < cov_max**2:
                need_learning = False
                need_enriching = False
            else:
                if cv_Pf > cov_max:
                    need_enriching = True
        else:
            need_learning = True

    return improve_criterion, need_learning, need_enriching


def calculate_importance_weight(x, original_dist, importance_dist):
    """
     Calculate the importance weight for a given point based on two multivariate normal distributions.
    Returns:
    importance_weight: The importance weight at the given point.
    """
    initial_pdf = original_dist.computePDF(x)
    importance_pdf = importance_dist.computePDF(x)

    # Avoid division by zero
    if importance_pdf == 0:
        return 0.0

    importance_weight = initial_pdf / importance_pdf
    return importance_weight


""" Main function including the Pf calculation and training of the metamodel"""


def Vb_AGP(m, sig, mu1, mu2, cov, performance_function, initial_distribution, distribution_chosen,
           initial_IS_sample_size, initial_doe_size=None, learning_function='EFF', cov_max=0.1, hot_start=False, path=''
           , corr='matern52'):
    var_G_E_X_l = list()
    var_X_E_G_l = list()
    var_tot_l = list()
    var_tot = 0.
    var_tot_inf = 0.
    var_tot_sup = 0.
    first_train_x = list()
    first_train_y = list()

    if hot_start == False:  # new initial DoE, new initial MC
        # initial MC population
        IS_sample_size = int(initial_IS_sample_size)
        ImportaneSamples = np.array(distribution_chosen.getSample(IS_sample_size))

        # initial DoE generation
        lhs = ot.MonteCarloExperiment(initial_distribution, initial_doe_size)
        DoE = np.array(lhs.generate())

        print("--------------------------------------")
        print("Evalulation of the perfomance function on the initial doe of size ", initial_doe_size)
        print("--------------------------------------")

        # Evaluation of the performance function - deflection of cantilever beam
        Y = np.zeros((initial_doe_size, 1))
        i = 0

        for x in DoE:
            Y[i, 0] = performance_function(x, m, sig)
            print(Y[i, 0])
            i = i + 1
        print("--------------------------------------")
        print("Saving the initial DoE ")
        print("--------------------------------------")
        ImportaneSamples.dump(path + str('ImportaneSamples_init.pk'))
        DoE.dump(path + str('DoE_init.pk'))
        Y.dump(path + str('Y_init.pk'))
        first_train_x = DoE
        first_train_y = Y

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
            Y[i, 0] = performance_function(x, m, sig)
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

    """
    Vb-AGP loop ------------->
    """

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
    sm = KPLS(eval_n_comp=True)
    sm.set_training_values(X_train, Y_train)
    sm.train()
    ncomp = sm.options["n_comp"]
    print("\n The model automatically choose " + str(ncomp) + " components.")

    cv_max = 0.05
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

        if i == 0:  # Storing first x and y Training input and variance of model JUST TO PLOT LATER
            first_model_x = ImportaneSamples
            first_model_y = y_glob_kpls
            first_model_var = KPLS_variance

        y_glob, MSE_glob = gp_g.predict(ImportaneSamples, eval_MSE=True)
        y_pred, MSE_pred = gp_g.predict(ImportaneSamples_candidates, eval_MSE=True)
        indicatrice_glob = np.zeros(np.size(y_glob_kpls))
        print("Importance Samples size:", np.size(ImportaneSamples, 0))

        # Weight calculation part
        weights = []
        norm_imp_weights = []

        for n in range(0, np.size(y_glob_kpls)):
            weights.append(calculate_importance_weight(ImportaneSamples[n], initial_distribution, distribution_chosen))

        for j in range(0, np.size(y_glob_kpls)):
            if y_glob_kpls[j] < 0:
                indicatrice_glob[j] = 1 * weights[j]

        Pf.append(np.sum(indicatrice_glob) / ImportaneSamples.shape[0])
        print("Pf=", Pf)

        if Pf[i] > 0.0:
            cv_Pf.append(np.sqrt(np.var(indicatrice_glob) / ImportaneSamples.shape[0] / Pf[i]))
            t2 = time.time()
            print("Pf = " + str(Pf[i]) + " CV = " + str(cv_Pf[i]) + " N_MC = " + str(
                ImportaneSamples.shape[0]) + " DOE = " + str(X_train.shape[0]) + " Time: " + str(t2 - t1))
            improve_criterion, need_learning, need_enriching = exploration(sig, y_glob_kpls, KPLS_variance, cv_Pf[i],
                                                                           stop=learning_function, cov_max=cv_max)
            print("Convergence max = " + str(improve_criterion.max()))

            if need_learning:  # Enriching DOE
                which_x_to_add = np.argmax(improve_criterion)
                val_star = improve_criterion[which_x_to_add]
                x_star = ImportaneSamples_candidates[which_x_to_add, :]
                ImportaneSamples_candidates = np.delete(ImportaneSamples_candidates, which_x_to_add, 0)
                y_star = performance_function(np.atleast_2d(x_star), m, sig)
                print("Enriching DOE: y star = ", y_star)
                X_train = np.vstack([X_train, np.atleast_2d(x_star)])
                Y_train = np.vstack([Y_train, y_star])
                #gp_g.fit(X_train, Y_train)
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

        elif Pf[i] == 0:
            cv_Pf.append(0)
            improve_criterion = EFF_criterion(y_glob_kpls, KPLS_variance)
            which_x_to_add = np.argmax(improve_criterion)
            x_star = ImportaneSamples_candidates[which_x_to_add, :]
            ImportaneSamples_candidates = np.delete(ImportaneSamples_candidates, which_x_to_add, 0)
            y_star = performance_function(np.atleast_2d(x_star), m, sig)
            print("Enriching DOE: y star = ", y_star)
            X_train = np.vstack([X_train, np.atleast_2d(x_star)])
            Y_train = np.vstack([Y_train, y_star])
            sm.set_training_values(X_train, Y_train)
            sm.train()

        n_iter += 1
        i += 1

    list_res.append(np.array([Pf, var_X_E_G_l, var_G_E_X_l, improve_criterion.max()], dtype=object))

    """ PLOTTING THE FIRST TRAINING OF THE KPLS MODEL """
    plt.plot(np.linspace(0, len(first_train_x), len(first_train_x)), first_train_y, "o")
    plt.plot(np.linspace(0, len(first_train_x), len(first_train_x)), first_model_y[0:initial_doe_size])
    plt.fill_between(
        np.ravel(np.linspace(0, len(first_train_x), len(first_train_x))),
        np.ravel(first_model_y[0:initial_doe_size] - 3 * np.sqrt(first_model_var[0:initial_doe_size])),
        np.ravel(first_model_y[0:initial_doe_size] + 3 * np.sqrt(first_model_var[0:initial_doe_size])),
        color="lightgrey",
    )
    plt.xlabel("Iteration")
    plt.ylabel("y")
    plt.title("First training of surrogate model")
    plt.legend(["Training data", "Prediction", "Confidence Interval 99%"])
    plt.show()

    last_train_x = X_train  # Storing last training points and last prediction points JUST TO PLOT SOON AFTERWARDS
    last_train_y = Y_train
    last_model_x = ImportaneSamples
    last_model_y = y_glob_kpls
    last_model_var = KPLS_variance

    """ PLOTTING THE LAST TRAINING OF THE KPLS MODEL """
    plt.plot(np.linspace(0, len(last_train_x), len(last_train_x)), last_train_y, "o")
    plt.plot(np.linspace(0, len(last_train_x), len(last_train_x)), last_model_y[0:len(last_train_x)])
    plt.fill_between(
        np.ravel(np.linspace(0, len(last_train_x), len(last_train_x))),
        np.ravel(last_model_y[0:len(last_train_x)] - 3 * np.sqrt(last_model_var[0:len(last_train_x)])),
        np.ravel(last_model_y[0:len(last_train_x)] + 3 * np.sqrt(last_model_var[0:len(last_train_x)])),
        color="lightgrey",
    )
    plt.xlabel("iteration")
    plt.ylabel("y")
    plt.title("Last training of surrogate model just before convergence")
    plt.legend(["Training data", "Prediction", "Confidence Interval 99%"])
    plt.show()

    return Pf, X_train, n_iter, gp_g, Y_train, ImportaneSamples.shape[0], cv_Pf, sm


def limitfunc(X, m, sig):
    G = m + (3 * sig * (m ** 0.5)) - np.sum(X)
    return G


"""Distributions"""

stochastic_dim = 40  # total number of dimensions of the problem
m = stochastic_dim
sig = 0.2  # std deviation of the distribution
temp_xvalues = np.ones(m)  # has no significance for the problem directly
path = ''

X_dist = ot.LogNormal()
X_dist.setParameter(ot.LogNormalMuSigma()([1, 0.2, 0.0]))
composed_dist = ot.ComposedDistribution([X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist,
                                         X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist,
                                         X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist,
                                         X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist]
                                        )

distribution = composed_dist
model = symboli_func(m, sig, temp_xvalues)


# IS additions

# Threshold event creation
vect = ot.RandomVector(distribution)
G = ot.CompositeRandomVector(model, vect)
event = ot.ThresholdEvent(G, ot.Less(), 0)

# FORM configuration
optimAlgo = ot.Cobyla()
optimAlgo.setMaximumEvaluationNumber(3000)
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

# Standard Gaussian centered around SPDP-standardSpaceDesignPoint
dimension = distribution.getDimension()
mu = np.array(distribution.getMean())  # mean of original composed distribution
cov = distribution.getCovariance()  # covariance of original composed distribution

importanceDistribution = ot.Normal(mu, cov)  # initialise importance distribution
importanceDistribution.setMean(physicalSpaceDesignPoint)  # finalisation of Importance density

fct_test = limitfunc
distrib_test = distribution
initial_doe_size = 30
initial_IS_sample_size = 1000
cov_max = 0.05

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

PF, DoE, n_iter, gp_g, Y, IS_sample_size, cov_tot, sm = Vb_AGP(m, sig, mu, physicalSpaceDesignPoint, cov, fct_test, distribution,
                                                           distrib_test, initial_IS_sample_size,
                                                           initial_doe_size=initial_doe_size, path=path,
                                                           cov_max=cov_max, corr='matern52')
end_time = time.time()
total_time = end_time - start_time


""" FINAL RESULTS PRINTING """

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


""" Pf VALUES PLOTTING """

plt.title('Failure Probability Vs Number of iterations of KPLS-IS')
plt.xlabel('Number of iterations of KPLS-IS')
plt.ylabel('Failure Probability (Pf)')
plt.plot(num_iter, PF, label="Mean")
plt.plot(num_iter, Pf_ub, label="Upper Bound")
plt.plot(num_iter, Pf_lb, label="Lower Bound")
plt.legend(loc="best")
plt.show()




