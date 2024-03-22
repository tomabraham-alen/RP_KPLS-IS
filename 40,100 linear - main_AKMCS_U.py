# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:48:35 2018


"""

import numpy as np
import scipy as sci
from scipy.stats import norm
import openturns as ot
from gaussian_process import GaussianProcess
from create_symbolic_function import symboli_func
from scipy import stats
import time
from openturns.usecases import cantilever_beam
import matplotlib.pyplot as plt

def U_criterion(y,MSE):
    return np.abs(y)/np.sqrt(MSE)

def EFF_criterion(y,MSE):
    # search of the best learning point x_star among the MC population using the EFF criterion 
    epsilon = 2.0*np.sqrt(MSE)
    X = y/np.sqrt(MSE)
    X1 = (-epsilon-y)/np.sqrt(MSE)
    X2 = (epsilon-y)/np.sqrt(MSE)
    X_cdf = sci.stats.norm.cdf(-X)
    X_pdf = sci.stats.norm.pdf(-X)
    X1_cdf = sci.stats.norm.cdf(X1)
    X1_pdf = sci.stats.norm.pdf(X1)
    X2_cdf = sci.stats.norm.cdf(X2)
    X2_pdf = sci.stats.norm.pdf(X2)
    EFF = y*(2.0*X_cdf-X1_cdf-X2_cdf)-np.sqrt(MSE)*(2.0*X_pdf-X1_pdf-X2_pdf)+epsilon*(X2_cdf-X1_cdf)
    return EFF

def exploration(y,MSE, cv_Pf, stop='EFF', cov_max = 0.05):
    need_learning = False
    need_enriching = False
    if stop=="EFF":
        improve_criterion = EFF_criterion(y,MSE)
        need_learning = improve_criterion.max()>1e-3
        need_enriching = cv_Pf>cov_max
    elif stop=="U":
        improve_criterion = -U_criterion(y,MSE)
        need_learning = improve_criterion.max()>-2
        need_enriching = cv_Pf>cov_max
    return improve_criterion, need_learning, need_enriching

def Vb_AGP(m, sig, performance_function,distribution_chosen,initial_MC_sample_size,initial_doe_size = None, learning_function = 'EFF',cov_max = 0.1, hot_start = False,path='',corr='matern52'):
    var_G_E_X_l=list()
    var_X_E_G_l=list()
    var_tot_l=list()
    var_tot=0.
    var_tot_inf=0.
    var_tot_sup=0.
    
    
    if hot_start == False: # new initial DoE, new initial MC
        #initial MC population
        MC_sample_size = int(initial_MC_sample_size)
        MC_sample = np.array(distribution_chosen.getSample(MC_sample_size))
        #print(MC_sample)
        # initial DoE generation
        initial_doe_size = int(initial_doe_size)
        lhs = ot.MonteCarloExperiment(distribution_chosen, initial_doe_size)
        DoE=np.array(lhs.generate())
        DoE_size = DoE.shape[0]
        # inital DoE evaluation 
        print("--------------------------------------")
        print("Evalulation of the perfomance function on the initial doe of size ",initial_doe_size)
        print("--------------------------------------")
       # Evaluation of the performance function
        Y = np.zeros((initial_doe_size,1))
        i = 0
        for x in DoE:
            Y[i,0] = performance_function(x, m, sig)
            print(Y[i,0])
            i = i+1
        print( "--------------------------------------")
        print("Saving the initial DoE ")
        print( "--------------------------------------")
        MC_sample.dump(path+str('MC_sample_init.pk'))
        DoE.dump(path+str('DoE_init.pk'))
        Y.dump(path+str('Y_init.pk'))
        
    elif hot_start == 'init': # reuse of the previous initial MC population and inital DoE
        MC_sample = np.load(path+str('MC_sample_init.pk'),allow_pickle=True)
        MC_sample_size = MC_sample.shape[0] 
        initial_MC_sample_size =MC_sample_size 
        DoE = np.load(path+str('DoE_init.pk'),allow_pickle=True)
        DoE_size = DoE.shape[0]
        Y = np.load(path+str('Y_init.pk'),allow_pickle=True)
        print( "--------------------------------------")
        print ("Reading the initial DoE of size ",DoE_size)
        print( "--------------------------------------")
 

    elif hot_start == 'init_doe': # new MC populaton and reuse of the initial DoE
        MC_sample_size = initial_MC_sample_size
        MC_sample = np.array(distribution_chosen.getSample(MC_sample_size))
        initial_MC_sample_size =MC_sample_size 
        DoE = np.load(path+str('DoE_init.pk'),allow_pickle=True)
        DoE_size = DoE.shape[0]
        Y = np.zeros((DoE_size,1))
        i = 0
        for x in DoE:
            Y[i,0] = performance_function(x, m, sig)
            print(Y[i,0])
            i = i+1
        print( "--------------------------------------")
        print ("Reading the initial DoE of size ",DoE_size)
        print( "--------------------------------------")

        
        
    elif hot_start == True: # reuse of the MC population and DoE (last update)
        MC_sample = np.load(path+str('MC_sample.pk'),allow_pickle=True)
        MC_sample_size = MC_sample.shape[0]  
        initial_MC_sample_size =MC_sample_size 
        DoE = np.load(path+str('DoE.pk'),allow_pickle=True)
        DoE_size = DoE.shape[0]
        Y = np.load(path+str('Y.pk'),allow_pickle=True)        
        print( "--------------------------------------")
        print ("Hot start with a DoE of size ",DoE_size)
        print( "--------------------------------------")
        
    #construction of the GP surrogate
    stochastic_dim = DoE.shape[1]    
    theta0 = np.array([0.1]*stochastic_dim)
    thetaL = np.array([1e-6]*stochastic_dim)
    thetaU = np.array([100.0]*stochastic_dim)
    gp_g = GaussianProcess(corr=corr,theta0 = theta0,thetaL=thetaL,thetaU=thetaU)     
    #Vb-AGP loop
    n_iter = 0
    convergence = False
    n_iter = 1
    Y_train = Y.copy()
    X_train = DoE.copy()    
    need_learning = True
    need_enriching = True
    MC_sample_candidates = MC_sample.copy()
    gp_g.fit(X_train, Y_train)
    cv_max = 0.05
    list_res = list()
    i = 0
    Pf = []
    cv_Pf = []
# In[15]:
    iter_max = 1000
    while n_iter<iter_max and (need_learning or need_enriching):
        print("______________________________________________________________________________")
        print ("iter =", n_iter)
        # Global prediction
        t1 = time.time()
        y_glob,MSE_glob = gp_g.predict(MC_sample,eval_MSE=True)
        y_pred,MSE_pred = gp_g.predict(MC_sample_candidates,eval_MSE=True)
        #print(np.transpose(y_pred))
        MSE_pred = MSE_pred.reshape(np.size(MSE_pred), 1)
        indicatrice_glob=np.zeros(np.size(y_glob))
        #print(indicatrice_glob)
        for j in range(0, np.size(y_glob)):
            if y_glob[j] < 1:
                indicatrice_glob[j] = 1

        #print("indicatrice_glob", indicatrice_glob)
        Pf.append(np.sum(indicatrice_glob)/MC_sample.shape[0])
        print("Pf=", Pf)

        if Pf[i] > 0.0:
            cv_Pf.append(np.sqrt((1-Pf[i])/MC_sample.shape[0]/Pf[i]))
            t2 = time.time()
            print("Pf = " + str(Pf[i]) + " CV = " + str(cv_Pf[i]) + " N_MC = " + str(MC_sample.shape[0]) + " DOE = " + str(X_train.shape[0]) + " Time: " + str(t2-t1))
            improve_criterion, need_learning, need_enriching = exploration(y_pred,MSE_pred,cv_Pf[i],stop=learning_function, cov_max=cv_max)
            print("Convergence max = " + str(improve_criterion.max()))
            if need_learning: #Enriching DOE
                which_x_to_add = np.argmax(improve_criterion)
                val_star = improve_criterion[which_x_to_add]
                x_star = MC_sample_candidates[which_x_to_add,:]
                MC_sample_candidates = np.delete(MC_sample_candidates, which_x_to_add, 0)
                y_star = performance_function(np.atleast_2d(x_star), m, sig)
                print("Enriching DOE: y star = ", y_star)
                X_train = np.vstack([X_train, np.atleast_2d(x_star)])
                Y_train = np.vstack([Y_train,y_star])
                gp_g.fit(X_train, Y_train)
            elif need_enriching: #Enriching MC
                new_MC = np.array(distribution_chosen.getSample(MC_sample_size))
                MC_sample_candidates = np.vstack([MC_sample_candidates, new_MC])
                MC_sample = np.vstack([MC_sample, new_MC])
                print("Enriching MC")
            else:
                print("Done")
        elif Pf[i] == 0:
            cv_Pf.append(0)
            improve_criterion = EFF_criterion(y_pred, MSE_pred)
            which_x_to_add = np.argmax(improve_criterion)
            x_star = MC_sample_candidates[which_x_to_add, :]
            MC_sample_candidates = np.delete(MC_sample_candidates, which_x_to_add, 0)
            y_star = performance_function(np.atleast_2d(x_star), m, sig)
            print("Enriching DOE: y star = ", y_star)
            X_train = np.vstack([X_train, np.atleast_2d(x_star)])
            Y_train = np.vstack([Y_train, y_star])
            gp_g.fit(X_train, Y_train)
            new_MC = np.array(distribution_chosen.getSample(MC_sample_size))
            # new_MC = new_MC.transpose()
            MC_sample_candidates = np.vstack([MC_sample_candidates, new_MC])
            MC_sample = np.vstack([MC_sample, new_MC])
            print("Enriching MC")
        n_iter += 1
        i = i + 1
    list_res.append(np.array([Pf,var_X_E_G_l, var_G_E_X_l,improve_criterion.max()], dtype=object))
    return Pf,X_train,n_iter,gp_g,Y_train,MC_sample.shape[0],cv_Pf

def limitfunc(X, m, sig):
    G = m + (3 * sig * (m ** 0.5)) - np.sum(X)
    return G

##### Distributions 

stochastic_dim = 40

m = stochastic_dim
sig = 0.2  # std deviation of the distribution
temp_xvalues = np.ones(m)  # has no significance for the problem directly
path = ''

X_dist = ot.LogNormal()
X_dist.setParameter(ot.LogNormalMuSigma()([1, 0.2, 0.0]))
composed_dist = ot.ComposedDistribution([X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist, X_dist])
distribution = composed_dist
model = symboli_func(m, sig, temp_xvalues)

fct_test = limitfunc
distrib_test=distribution
initial_doe_size= 30
initial_MC_sample_size=1000
cov_max=0.05

Pf=[]
Pf_ub = []
Pf_lb = []
var = []

start_time = time.time()
#for i in initial_doe_size:
PF, DoE, n_iter, gp_g, Y, MC_sample_size, cov_tot = Vb_AGP(m, sig, fct_test, distrib_test, initial_MC_sample_size, initial_doe_size, path=path, cov_max=cov_max, corr='matern52')
#Pf.append(PF)
end_time = time.time()
total_time = end_time - start_time

#plt.title('Failure Probability Vs DoE size')
#plt.xlabel('DoE Size')
#plt.ylabel('Failure Probability (Pf)')
#plt.ylim(0.01, 0.05)
#plt.plot(initial_doe_size, Pf)
#plt.show()

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

print(f"Total time taken is {total_time:.4f} seconds")
plt.title('Failure Probability Vs Number of iterations of AK-MCS')
plt.xlabel('Number of iterations of AK-MCS')
plt.ylabel('Failure Probability (Pf)')
# plt.ylim(0.025, 0.05)
plt.plot(num_iter, PF, label="Mean")
plt.plot(num_iter, Pf_ub, label="Upper Bound")
plt.plot(num_iter, Pf_lb, label="Lower Bound")
plt.legend(loc="best")
plt.show()
	 
