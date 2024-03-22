# -*- coding: utf-8 -*-
# Authors: Vincent Dubourg <vincent.dubourg@gmail.com>
#         (mostly translation, see implementation details)
# Licence: BSD 3 clause
# 4 functions add by onera : conditioned_covariance_function
#                            random_trajectories
#                            KL_expansion
#                            eval_KL_vect

# pragma: no cover
from __future__ import print_function

import numpy as np
from scipy import linalg, optimize
from scipy.stats import norm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.utils import check_random_state, check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
#from . import regression_models as regression
#from . import correlation_models as correlation
import regression_models as regression
import correlation_models as correlation
import openturns as ot 
MACHINE_EPSILON = np.finfo(np.double).eps
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import _sparse_random_matrix

def l1_cross_distances(X):  # pragma: no cover
    """
    Computes the nonzero componentwise L1 cross-distances between the vectors
    in X.

    Parameters
    ----------

    X: array_like
        An array with shape (n_samples, n_features)

    Returns
    -------

    D: array with shape (n_samples * (n_samples - 1) / 2, n_features)
        The array of componentwise L1 cross-distances.

    ij: arrays with shape (n_samples * (n_samples - 1) / 2, 2)
        The indices i and j of the vectors in X associated to the cross-
        distances in D: D[k] = np.abs(X[ij[k, 0]] - Y[ij[k, 1]]).
    """
    X = check_array(X)
    n_samples, n_features = X.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=int)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0
    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = np.abs(X[k] - X[(k + 1):n_samples])

    return D, ij


class GaussianProcess(BaseEstimator, RegressorMixin):  # pragma: no cover
    """The Gaussian Process model class.

    Read more in the :ref:`User Guide <gaussian_process>`.

    Parameters
    ----------
    regr : string or callable, optional
        A regression function returning an array of outputs of the linear
        regression functional basis. The number of observations n_samples
        should be greater than the size p of this basis.
        Default assumes a simple constant regression trend.
        Available built-in regression models are::

            'constant', 'linear', 'quadratic'

    corr : string or callable, optional
        A stationary autocorrelation function returning the autocorrelation
        between two points x and x'.
        Default assumes a squared-exponential autocorrelation model.
        Built-in correlation models are::

            'absolute_exponential', 'squared_exponential',
            'generalized_exponential', 'cubic', 'linear'

    beta0 : double array_like, optional
        The regression weight vector to perform Ordinary Kriging (OK).
        Default assumes Universal Kriging (UK) so that the vector beta of
        regression weights is estimated using the maximum likelihood
        principle.

    storage_mode : string, optional
        A string specifying whether the Cholesky decomposition of the
        correlation matrix should be stored in the class (storage_mode =
        'full') or not (storage_mode = 'light').
        Default assumes storage_mode = 'full', so that the
        Cholesky decomposition of the correlation matrix is stored.
        This might be a useful parameter when one is not interested in the
        MSE and only plan to estimate the BLUP, for which the correlation
        matrix is not required.

    verbose : boolean, optional
        A boolean specifying the verbose level.
        Default is verbose = False.

    theta0 : double array_like, optional
        An array with shape (n_features, ) or (1, ).
        The parameters in the autocorrelation model.
        If thetaL and thetaU are also specified, theta0 is considered as
        the starting point for the maximum likelihood estimation of the
        best set of parameters.
        Default assumes isotropic autocorrelation model with theta0 = 1e-1.

    thetaL : double array_like, optional
        An array with shape matching theta0's.
        Lower bound on the autocorrelation parameters for maximum
        likelihood estimation.
        Default is None, so that it skips maximum likelihood estimation and
        it uses theta0.

    thetaU : double array_like, optional
        An array with shape matching theta0's.
        Upper bound on the autocorrelation parameters for maximum
        likelihood estimation.
        Default is None, so that it skips maximum likelihood estimation and
        it uses theta0.

    normalize : boolean, optional
        Input X and observations y are centered and reduced wrt
        means and standard deviations estimated from the n_samples
        observations provided.
        Default is normalize = True so that data is normalized to ease
        maximum likelihood estimation.

    nugget : double or ndarray, optional
        Introduce a nugget effect to allow smooth predictions from noisy
        data.  If nugget is an ndarray, it must be the same length as the
        number of data points used for the fit.
        The nugget is added to the diagonal of the assumed training covariance;
        in this way it acts as a Tikhonov regularization in the problem.  In
        the special case of the squared exponential correlation function, the
        nugget mathematically represents the variance of the input values.
        Default assumes a nugget close to machine precision for the sake of
        robustness (nugget = 10. * MACHINE_EPSILON).

    optimizer : string, optional
        A string specifying the optimization algorithm to be used.
        Default uses 'fmin_cobyla' algorithm from scipy.optimize.
        Available optimizers are::

            'fmin_cobyla', 'Welch'

        'Welch' optimizer is dued to Welch et al., see reference [WBSWM1992]_.
        It consists in iterating over several one-dimensional optimizations
        instead of running one single multi-dimensional optimization.

    random_start : int, optional
        The number of times the Maximum Likelihood Estimation should be
        performed from a random starting point.
        The first MLE always uses the specified starting point (theta0),
        the next starting points are picked at random according to an
        exponential distribution (log-uniform on [thetaL, thetaU]).
        Default does not use random starting point (random_start = 1).

    random_state: integer or numpy.RandomState, optional
        The generator used to shuffle the sequence of coordinates of theta in
        the Welch optimizer. If an integer is given, it fixes the seed.
        Defaults to the global numpy random number generator.


    Attributes
    ----------
    theta_ : array
        Specified theta OR the best set of autocorrelation parameters (the \
        sought maximizer of the reduced likelihood function).

    reduced_likelihood_function_value_ : array
        The optimal reduced likelihood function value.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.gaussian_process import GaussianProcess
    >>> X = np.array([[1., 3., 5., 6., 7., 8.]]).T
    >>> y = (X * np.sin(X)).ravel()
    >>> gp = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1.)
    >>> gp.fit(X, y)                                      # doctest: +ELLIPSIS
    GaussianProcess(beta0=None...
            ...

    Notes
    -----
    The presentation implementation is based on a translation of the DACE
    Matlab toolbox, see reference [NLNS2002]_.

    References
    ----------

    .. [NLNS2002] `H.B. Nielsen, S.N. Lophaven, H. B. Nielsen and J.
        Sondergaard.  DACE - A MATLAB Kriging Toolbox.` (2002)
        http://www2.imm.dtu.dk/~hbn/dace/dace.pdf

    .. [WBSWM1992] `W.J. Welch, R.J. Buck, J. Sacks, H.P. Wynn, T.J. Mitchell,
        and M.D.  Morris (1992). Screening, predicting, and computer
        experiments.  Technometrics, 34(1) 15--25.`
        http://www.jstor.org/pss/1269548
    """

    _regression_types = {
        'constant': regression.constant,
        'linear': regression.linear,
        'quadratic': regression.quadratic}

    _correlation_types = {
        'matern32': correlation.matern32,
        'matern52': correlation.matern52,
        'absolute_exponential': correlation.absolute_exponential,
        'squared_exponential': correlation.squared_exponential,
        'generalized_exponential': correlation.generalized_exponential,
        'cubic': correlation.cubic,
        'linear': correlation.linear}

    _optimizer_types = [
        'fmin_cobyla',
        'Welch']

    def __init__(self, regr='constant', corr='squared_exponential', beta0=None,
                 storage_mode='full', verbose=False, theta0=1e-1,
                 thetaL=None, thetaU=None, optimizer='fmin_cobyla',
                 random_start=1, normalize=True,
                 nugget=10. * MACHINE_EPSILON, random_state=None):

        self.regr = regr
        self.corr = corr
        self.beta0 = beta0
        self.storage_mode = storage_mode
        self.verbose = verbose
        self.theta0 = theta0
        self.thetaL = thetaL
        self.thetaU = thetaU
        self.normalize = normalize
        self.nugget = nugget
        self.optimizer = optimizer
        self.random_start = random_start
        self.random_state = random_state

    def fit(self, X, y):
        """
        The Gaussian Process model fitting method.

        Parameters
        ----------
        X : double array_like
            An array with shape (n_samples, n_features) with the input at which
            observations were made.

        y : double array_like
            An array with shape (n_samples, ) or shape (n_samples, n_targets)
            with the observations of the output to be predicted.

        Returns
        -------
        gp : self
            A fitted Gaussian Process model object awaiting data to perform
            predictions.
        """
        # Run input checks
        self._check_params()

        self.random_state = check_random_state(self.random_state)

        # Force data to 2D numpy.array
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        self.y_ndim_ = y.ndim
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # Check shapes of DOE & observations
        n_samples, n_features = X.shape
        _, n_targets = y.shape

        # Run input checks
        self._check_params(n_samples)

        # Normalize data or don't
        if self.normalize:
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            y_mean = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            X_std[X_std == 0.] = 1.
            y_std[y_std == 0.] = 1.
            # center and scale X if necessary
            X = (X - X_mean) / X_std
            y = (y - y_mean) / y_std
        else:
            X_mean = np.zeros(1)
            X_std = np.ones(1)
            y_mean = np.zeros(1)
            y_std = np.ones(1)

        # Calculate matrix of distances D between samples
        D, ij = l1_cross_distances(X)
        if (np.min(np.sum(D, axis=1)) == 0.
                and self.corr != correlation.pure_nugget):
            raise Exception("Multiple input features cannot have the same"
                            " target value.")

        # Regression matrix and parameters
        F = self.regr(X)
        n_samples_F = F.shape[0]
        if F.ndim > 1:
            p = F.shape[1]
        else:
            p = 1
        if n_samples_F != n_samples:
            raise Exception("Number of rows in F and X do not match. Most "
                            "likely something is going wrong with the "
                            "regression model.")
        if p > n_samples_F:
            raise Exception(("Ordinary least squares problem is undetermined "
                             "n_samples=%d must be greater than the "
                             "regression model size p=%d.") % (n_samples_F, p))
        if self.beta0 is not None:
            if self.beta0.shape[0] != p:
                raise Exception("Shapes of beta0 and F do not match.")

        # Set attributes
        self.X = X
        self.y = y
        self.D = D
        self.ij = ij
        self.F = F
        self.X_mean, self.X_std = X_mean, X_std
        self.y_mean, self.y_std = y_mean, y_std

        # Determine Gaussian Process model parameters
        if self.thetaL is not None and self.thetaU is not None:
            # Maximum Likelihood Estimation of the parameters
            if self.verbose:
                print("Performing Maximum Likelihood Estimation of the "
                      "autocorrelation parameters...")
            self.theta_, self.reduced_likelihood_function_value_, par = \
                self._arg_max_reduced_likelihood_function()
            if np.isinf(self.reduced_likelihood_function_value_):
                raise Exception("Bad parameter region. "
                                "Try increasing upper bound")

        else:
            # Given parameters
            if self.verbose:
                print("Given autocorrelation parameters. "
                      "Computing Gaussian Process model parameters...")
            self.theta_ = self.theta0
            self.reduced_likelihood_function_value_, par = \
                self.reduced_likelihood_function()
            if np.isinf(self.reduced_likelihood_function_value_):
                raise Exception("Bad point. Try increasing theta0.")

        self.beta = par['beta']
        self.gamma = par['gamma']
        self.sigma2 = par['sigma2']
        self.C = par['C']
        self.Ft = par['Ft']
        self.G = par['G']

        if self.storage_mode == 'light':
            # Delete heavy data (it will be computed again if required)
            # (it is required only when MSE is wanted in self.predict)
            if self.verbose:
                print("Light storage mode specified. "
                      "Flushing autocorrelation matrix...")
            self.D = None
            self.ij = None
            self.F = None
            self.C = None
            self.Ft = None
            self.G = None

        return self

    def predict(self, X, eval_MSE=False, batch_size=None):
        """
        This function evaluates the Gaussian Process model at x.

        Parameters
        ----------
        X : array_like
            An array with shape (n_eval, n_features) giving the point(s) at
            which the prediction(s) should be made.

        eval_MSE : boolean, optional
            A boolean specifying whether the Mean Squared Error should be
            evaluated or not.
            Default assumes evalMSE = False and evaluates only the BLUP (mean
            prediction).

        batch_size : integer, optional
            An integer giving the maximum number of points that can be
            evaluated simultaneously (depending on the available memory).
            Default is None so that all given points are evaluated at the same
            time.

        Returns
        -------
        y : array_like, shape (n_samples, ) or (n_samples, n_targets)
            An array with shape (n_eval, ) if the Gaussian Process was trained
            on an array of shape (n_samples, ) or an array with shape
            (n_eval, n_targets) if the Gaussian Process was trained on an array
            of shape (n_samples, n_targets) with the Best Linear Unbiased
            Prediction at x.

        MSE : array_like, optional (if eval_MSE == True)
            An array with shape (n_eval, ) or (n_eval, n_targets) as with y,
            with the Mean Squared Error at x.
        """
        check_is_fitted(self, "X")

        # Check input shapes
        X = check_array(X)
        n_eval, _ = X.shape
        n_samples, n_features = self.X.shape
        n_samples_y, n_targets = self.y.shape

        # Run input checks
        self._check_params(n_samples)


        if X.shape[1] != n_features:
            raise ValueError(("The number of features in X (X.shape[1] = %d) "
                              "should match the number of features used "
                              "for fit() "
                              "which is %d.") % (X.shape[1], n_features))

        if batch_size is None:
            # No memory management
            # (evaluates all given points in a single batch run)

            # Normalize input
            X = (X - self.X_mean) / self.X_std

            # Initialize output
            y = np.zeros(n_eval)
            if eval_MSE:
                MSE = np.zeros(n_eval)

            # Get pairwise componentwise L1-distances to the input training set
            dx = manhattan_distances(X, Y=self.X, sum_over_features=False)
            # Get regression function and correlation
            f = self.regr(X)
            r = self.corr(self.theta_, dx).reshape(n_eval, n_samples)

            # Scaled predictor
            y_ = np.dot(f, self.beta) + np.dot(r, self.gamma)

            # Predictor
            y = (self.y_mean + self.y_std * y_).reshape(n_eval, n_targets)

            if self.y_ndim_ == 1:
                y = y.ravel()

            # Mean Squared Error
            if eval_MSE:
                C = self.C
                if C is None:
                    # Light storage mode (need to recompute C, F, Ft and G)
                    if self.verbose:
                        print("This GaussianProcess used 'light' storage mode "
                              "at instantiation. Need to recompute "
                              "autocorrelation matrix...")
                    reduced_likelihood_function_value, par = \
                        self.reduced_likelihood_function()
                    self.C = par['C']
                    self.Ft = par['Ft']
                    self.G = par['G']

                rt = linalg.solve_triangular(self.C, r.T, lower=True)

                if self.beta0 is None:
                    # Universal Kriging
                    u = linalg.solve_triangular(self.G.T,
                                                np.dot(self.Ft.T, rt) - f.T,
                                                lower=True)
                else:
                    # Ordinary Kriging
                    u = np.zeros((n_targets, n_eval))

                MSE = np.dot(self.sigma2.reshape(n_targets, 1),
                             (1. - (rt ** 2.).sum(axis=0)
                              + (u ** 2.).sum(axis=0))[np.newaxis, :])
                MSE = np.sqrt((MSE ** 2.).sum(axis=0) / n_targets)

                # Mean Squared Error might be slightly negative depending on
                # machine precision: force to zero!
                MSE[MSE < 0.] = 0.

                if self.y_ndim_ == 1:
                    MSE = MSE.ravel()

                return y, MSE

            else:

                return y

        else:
            # Memory management

            if type(batch_size) is not int or batch_size <= 0:
                raise Exception("batch_size must be a positive integer")

            if eval_MSE:

                y, MSE = np.zeros(n_eval), np.zeros(n_eval)
                for k in range(max(1, n_eval / batch_size)):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size + 1, n_eval + 1])
                    y[batch_from:batch_to], MSE[batch_from:batch_to] = \
                        self.predict(X[batch_from:batch_to],
                                     eval_MSE=eval_MSE, batch_size=None)

                return y, MSE

            else:

                y = np.zeros(n_eval)
                for k in range(max(1, n_eval / batch_size)):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size + 1, n_eval + 1])
                    y[batch_from:batch_to] = \
                        self.predict(X[batch_from:batch_to],
                                     eval_MSE=eval_MSE, batch_size=None)

                return y

    def reduced_likelihood_function(self, theta=None):
        """
        This function determines the BLUP parameters and evaluates the reduced
        likelihood function for the given autocorrelation parameters theta.

        Maximizing this function wrt the autocorrelation parameters theta is
        equivalent to maximizing the likelihood of the assumed joint Gaussian
        distribution of the observations y evaluated onto the design of
        experiments X.

        Parameters
        ----------
        theta : array_like, optional
            An array containing the autocorrelation parameters at which the
            Gaussian Process model parameters should be determined.
            Default uses the built-in autocorrelation parameters
            (ie ``theta = self.theta_``).

        Returns
        -------
        reduced_likelihood_function_value : double
            The value of the reduced likelihood function associated to the
            given autocorrelation parameters theta.

        par : dict
            A dictionary containing the requested Gaussian Process model
            parameters:

                sigma2
                        Gaussian Process variance.
                beta
                        Generalized least-squares regression weights for
                        Universal Kriging or given beta0 for Ordinary
                        Kriging.
                gamma
                        Gaussian Process weights.
                C
                        Cholesky decomposition of the correlation matrix [R].
                Ft
                        Solution of the linear equation system : [R] x Ft = F
                G
                        QR decomposition of the matrix Ft.
        """
        check_is_fitted(self, "X")

        if theta is None:
            # Use built-in autocorrelation parameters
            theta = self.theta_

        # Initialize output
        reduced_likelihood_function_value = - np.inf
        par = {}

        # Retrieve data
        n_samples = self.X.shape[0]
        D = self.D
        ij = self.ij
        F = self.F

        if D is None:
            # Light storage mode (need to recompute D, ij and F)
            D, ij = l1_cross_distances(self.X)
            if (np.min(np.sum(D, axis=1)) == 0.
                    and self.corr != correlation.pure_nugget):
                raise Exception("Multiple X are not allowed")
            F = self.regr(self.X)

        # Set up R
        r = self.corr(theta, D)
        R = np.eye(n_samples) * (1. + self.nugget)
        R[ij[:, 0], ij[:, 1]] = r
        R[ij[:, 1], ij[:, 0]] = r

        # Cholesky decomposition of R
        try:
            C = linalg.cholesky(R, lower=True)
        except linalg.LinAlgError:
            return reduced_likelihood_function_value, par

        # Get generalized least squares solution
        Ft = linalg.solve_triangular(C, F, lower=True)
        try:
            Q, G = linalg.qr(Ft, econ=True)
        except:
            #/usr/lib/python2.6/dist-packages/scipy/linalg/decomp.py:1177:
            # DeprecationWarning: qr econ argument will be removed after scipy
            # 0.7. The economy transform will then be available through the
            # mode='economic' argument.
            Q, G = linalg.qr(Ft, mode='economic')
            pass

        sv = linalg.svd(G, compute_uv=False)
        rcondG = sv[-1] / sv[0]
        if rcondG < 1e-10:
            # Check F
            sv = linalg.svd(F, compute_uv=False)
            condF = sv[0] / sv[-1]
            if condF > 1e15:
                raise Exception("F is too ill conditioned. Poor combination "
                                "of regression model and observations.")
            else:
                # Ft is too ill conditioned, get out (try different theta)
                return reduced_likelihood_function_value, par
        self.C_=C
        Yt = linalg.solve_triangular(C, self.y, lower=True)
        if self.beta0 is None:
            # Universal Kriging
            beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))
        else:
            # Ordinary Kriging
            beta = np.array(self.beta0)
        rho = Yt - np.dot(Ft, beta)
        sigma2 = (rho ** 2.).sum(axis=0) / n_samples
        # The determinant of R is equal to the squared product of the diagonal
        # elements of its Cholesky decomposition C
        detR = (np.diag(C) ** (2. / n_samples)).prod()

        # Compute/Organize output
        reduced_likelihood_function_value = - sigma2.sum() * detR
        par['sigma2'] = sigma2 * self.y_std ** 2.
        par['beta'] = beta
        par['gamma'] = linalg.solve_triangular(C.T, rho)
        par['C'] = C
        par['Ft'] = Ft
        par['G'] = G
        self.G_=G
        self.Q_=Q
       
        self.Ft_=Ft
        self.Yt=Yt
        
        

        return reduced_likelihood_function_value, par

    def _arg_max_reduced_likelihood_function(self):
        """
        This function estimates the autocorrelation parameters theta as the
        maximizer of the reduced likelihood function.
        (Minimization of the opposite reduced likelihood function is used for
        convenience)

        Parameters
        ----------
        self : All parameters are stored in the Gaussian Process model object.

        Returns
        -------
        optimal_theta : array_like
            The best set of autocorrelation parameters (the sought maximizer of
            the reduced likelihood function).

        optimal_reduced_likelihood_function_value : double
            The optimal reduced likelihood function value.

        optimal_par : dict
            The BLUP parameters associated to thetaOpt.
        """

        # Initialize output
        best_optimal_theta = []
        best_optimal_rlf_value = []
        best_optimal_par = []

        if self.verbose:
            print("The chosen optimizer is: " + str(self.optimizer))
            if self.random_start > 1:
                print(str(self.random_start) + " random starts are required.")

        percent_completed = 0.

        # Force optimizer to fmin_cobyla if the model is meant to be isotropic
        if self.optimizer == 'Welch' and self.theta0.size == 1:
            self.optimizer = 'fmin_cobyla'

        if self.optimizer == 'fmin_cobyla':

            def minus_reduced_likelihood_function(log10t):
                return - self.reduced_likelihood_function(
                    theta=10. ** log10t)[0]

            constraints = []
            for i in range(self.theta0.size):
                """ Old version for scipy < 0.19
                constraints.append(lambda log10t, i=i:
                                   log10t[i] - np.log10(self.thetaL[0, i]))
                constraints.append(lambda log10t, i=i:
                                   np.log10(self.thetaU[0, i]) - log10t[i])
                """
                # New version for scipy > 0.19
                constraints.append({'type' : 'ineq', 'fun': lambda log10t, i=i:
                                   log10t[i] - np.log10(self.thetaL[0, i])})
                constraints.append({'type' : 'ineq', 'fun':lambda log10t, i=i:
                                   np.log10(self.thetaU[0, i]) - log10t[i]})
            for k in range(self.random_start):
                if k == 0:
                    # Use specified starting point as first guess
                    theta0 = self.theta0
                else:
                    # Generate a random starting point log10-uniformly
                    # distributed between bounds
                    log10theta0 = (np.log10(self.thetaL)
                                   + self.random_state.rand(*self.theta0.shape)
                                   * np.log10(self.thetaU / self.thetaL))
                    theta0 = 10. ** log10theta0

                # Run Cobyla
                try:
                    """ Old version of scipy < 0.19
                    log10_optimal_theta = \
                        optimize.fmin_cobyla(minus_reduced_likelihood_function,
                                             np.log10(
                                                 theta0).ravel(), constraints,
                                             iprint=0)
                    """
                    # New version of scipy > 0.19
                    log10_optimal_theta = optimize.minimize(minus_reduced_likelihood_function,np.log10(theta0).ravel(),\
                            method='COBYLA', constraints=constraints)
                except ValueError as ve:
                    print("Optimization failed. Try increasing the ``nugget``")
                    raise ve
                """ Old version for scipy < 0.19
                optimal_theta = 10. ** log10_optimal_theta
                """
                # New version for scipy > 0.19
                optimal_theta = 10. ** log10_optimal_theta.x
                optimal_rlf_value, optimal_par = \
                    self.reduced_likelihood_function(theta=optimal_theta)

                # Compare the new optimizer to the best previous one
                if k > 0:
                    if optimal_rlf_value > best_optimal_rlf_value:
                        best_optimal_rlf_value = optimal_rlf_value
                        best_optimal_par = optimal_par
                        best_optimal_theta = optimal_theta
                else:
                    best_optimal_rlf_value = optimal_rlf_value
                    best_optimal_par = optimal_par
                    best_optimal_theta = optimal_theta
                if self.verbose and self.random_start > 1:
                    if (20 * k) / self.random_start > percent_completed:
                        percent_completed = (20 * k) / self.random_start
                        print("%s completed" % (5 * percent_completed))

            optimal_rlf_value = best_optimal_rlf_value
            optimal_par = best_optimal_par
            optimal_theta = best_optimal_theta

        elif self.optimizer == 'Welch':

            # Backup of the given atrributes
            theta0, thetaL, thetaU = self.theta0, self.thetaL, self.thetaU
            corr = self.corr
            verbose = self.verbose

            # This will iterate over fmin_cobyla optimizer
            self.optimizer = 'fmin_cobyla'
            self.verbose = False

            # Initialize under isotropy assumption
            if verbose:
                print("Initialize under isotropy assumption...")
            self.theta0 = check_array(self.theta0.min())
            self.thetaL = check_array(self.thetaL.min())
            self.thetaU = check_array(self.thetaU.max())
            theta_iso, optimal_rlf_value_iso, par_iso = \
                self._arg_max_reduced_likelihood_function()
            optimal_theta = theta_iso + np.zeros(theta0.shape)

            # Iterate over all dimensions of theta allowing for anisotropy
            if verbose:
                print("Now improving allowing for anisotropy...")
            for i in self.random_state.permutation(theta0.size):
                if verbose:
                    print("Proceeding along dimension %d..." % (i + 1))
                self.theta0 = check_array(theta_iso)
                self.thetaL = check_array(thetaL[0, i])
                self.thetaU = check_array(thetaU[0, i])

                def corr_cut(t, d):
                    return corr(check_array(np.hstack([optimal_theta[0][0:i],
                                                       t[0],
                                                       optimal_theta[0][(i +
                                                                         1)::]])),
                                d)

                self.corr = corr_cut
                optimal_theta[0, i], optimal_rlf_value, optimal_par = \
                    self._arg_max_reduced_likelihood_function()

            # Restore the given atrributes
            self.theta0, self.thetaL, self.thetaU = theta0, thetaL, thetaU
            self.corr = corr
            self.optimizer = 'Welch'
            self.verbose = verbose

        else:

            raise NotImplementedError("This optimizer ('%s') is not "
                                      "implemented yet. Please contribute!"
                                      % self.optimizer)

        return optimal_theta, optimal_rlf_value, optimal_par

    def _check_params(self, n_samples=None):

        # Check regression model
        if not callable(self.regr):
            if self.regr in self._regression_types:
                self.regr = self._regression_types[self.regr]
            else:
                raise ValueError("regr should be one of %s or callable, "
                                 "%s was given."
                                 % (self._regression_types.keys(), self.regr))

        # Check regression weights if given (Ordinary Kriging)
        if self.beta0 is not None:
            self.beta0 = np.atleast_2d(self.beta0)
            if self.beta0.shape[1] != 1:
                # Force to column vector
                self.beta0 = self.beta0.T

        # Check correlation model
        if not callable(self.corr):
            if self.corr in self._correlation_types:
                self.corr = self._correlation_types[self.corr]
            else:
                raise ValueError("corr should be one of %s or callable, "
                                 "%s was given."
                                 % (self._correlation_types.keys(), self.corr))

        # Check storage mode
        if self.storage_mode != 'full' and self.storage_mode != 'light':
            raise ValueError("Storage mode should either be 'full' or "
                             "'light', %s was given." % self.storage_mode)

        # Check correlation parameters
        self.theta0 = np.atleast_2d(self.theta0)
        lth = self.theta0.size

        if self.thetaL is not None and self.thetaU is not None:
            self.thetaL = np.atleast_2d(self.thetaL)
            self.thetaU = np.atleast_2d(self.thetaU)
            if self.thetaL.size != lth or self.thetaU.size != lth:
                raise ValueError("theta0, thetaL and thetaU must have the "
                                 "same length.")
            if np.any(self.thetaL <= 0) or np.any(self.thetaU < self.thetaL):
                raise ValueError("The bounds must satisfy O < thetaL <= "
                                 "thetaU.")

        elif self.thetaL is None and self.thetaU is None:
            if np.any(self.theta0 <= 0):
                raise ValueError("theta0 must be strictly positive.")

        elif self.thetaL is None or self.thetaU is None:
            raise ValueError("thetaL and thetaU should either be both or "
                             "neither specified.")

        # Force verbose type to bool
        self.verbose = bool(self.verbose)

        # Force normalize type to bool
        self.normalize = bool(self.normalize)

        # Check nugget value
        self.nugget = np.asarray(self.nugget)
        if np.any(self.nugget) < 0.:
            raise ValueError("nugget must be positive or zero.")
        if (n_samples is not None
                and self.nugget.shape not in [(), (n_samples,)]):
            raise ValueError("nugget must be either a scalar "
                             "or array of length n_samples.")

        # Check optimizer
        if self.optimizer not in self._optimizer_types:
            raise ValueError("optimizer should be one of %s"
                             % self._optimizer_types)

        # Force random_start type to int
        self.random_start = int(self.random_start)

    def conditioned_covariance_function(self, X):
        """
        This function computes the conditioned covariance function at points X.

        Parameters
        ----------
        K : array_like
            Values of the conditioned covariance function at points X

        self :	All parameters are stored in the Gaussian Process model object.
        X : array_like
            An array with shape (n_eval, n_features) giving the point(s) at
            which the conditioned covariance should be computed.

        Returns
        -------
                """
        n_eval, _ = X.shape
        n_samples, n_features = self.X.shape
        n_samples_y, n_targets = self.y.shape
        C = self.C
        Ft = self.Ft
        if C is None:
            # Light storage mode (need to recompute C, F, Ft and G)
            if self.verbose:
                print("This GaussianProcess used 'light' storage mode "
                      "at instantiation. Need to recompute "
                      "autocorrelation matrix...")
                reduced_likelihood_function_value, par = \
                    self.reduced_likelihood_function()
                self.C = par['C']
                self.Ft = par['Ft']
                self.G = par['G']
           # Normalize input
        if self.normalize == True:
            X = (X - self.X_mean) / self.X_std
        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(X, Y=self.X, sum_over_features=False)
        # Get regression function and correlation
        f = self.regr(X)
        r = self.corr(self.theta_, dx).reshape(n_eval, n_samples)
        # Get correlation matrix at new points
        D_new, ij = l1_cross_distances(X)
        # Set up R_new
        r_new = self.corr(self.theta_, D_new)
        R_new = np.eye(n_eval) * (1. + self.nugget)
        R_new[ij[:, 0], ij[:, 1]] = r_new
        R_new[ij[:, 1], ij[:, 0]] = r_new
        # Compute conditioned covariance matrix
        rt = linalg.solve_triangular(C, r.T, lower=True)
        M = np.dot(rt.T, rt)
        M[M > 1.0] = 1.0
        if self.beta0 is None:
            # Universal Kriging
            T1 = f - np.dot(rt.T, Ft)
            T2 = linalg.inv(np.dot(Ft.T, Ft))
            T3 = np.dot(T2, T1.T)
            M2 = np.dot(T1, T3)
            K = self.sigma2*(R_new - M + M2)
        else:
            # Ordinary Kriging
            K =self.sigma2*(R_new - M)
        self.K = K
        return K

    def random_trajectories(self, X, n,random_state=None):
        """
        This function computes random trajectories of the GP discretized at points X.

        Parameters
        ----------
        self :	All parameters are stored in the Gaussian Process model object.
        X : array_like
            An array with shape (n_eval, n_features) giving the point(s) at
            which the random trajectories are discretized.
        n : int
            Number of random trajectories to compute

        Returns
        -------
        Traj : array_like
            Values of the random trajectories discretized at points X

                """
        # Computation of the conditioned covariance matrix at X
        K= self.conditioned_covariance_function(X)
        # Cholesky factorization of K
        L_n = linalg.cholesky(K, lower=True)
        # Mean prediction
        mean = self.predict(X)
        # Trajectories
        Traj = []
        x_norm=norm()
        xi = x_norm.rvs((len(X),n),random_state=random_state)
        for i in range(n):
            xi_i=np.atleast_2d(xi[:,i]).T
            Traj.append(mean + np.dot(L_n, xi_i))
        Traj = np.array(Traj)
        return Traj[:,:,0]


    def K_eig_noncond_grid(self, eps, Omega,normalize_Omega=False,n_start=11):
        """
        This function computes the eigenvalues and the eigenvectors of B=sqrt(W) * C * sqrt(W) using the Nyström method with a uniform equi-spaced grid of Omega + Simpson quadrature formula
        C : the non conditioned co variance matrix for the PCA
        Parameters
        ----------
        self :	All parameters are stored in the Gaussian Process model object.
        eps : float
            Tolerance to achieve for the truncature of the KL expansion
        Omega : array_like
            Integration domain (normalized or not) boundaries (n_features,2) 
        normalize_Omega : boolean
                if False : Omega already normalize : no action 
                elif True : normalization of Omega 
        n_start : int 
            nb of intern integrations levels. n must be even.
        Returns
        -------
        
        eig : eigenvalues
        vect_star : eigenvectors
        nodes : integration nodes  
        weights : integration weights
        M : nb of retained eigenvalues 

        """
        if normalize_Omega : 
            Omega= (Omega - self.X_mean) / self.X_std

        
        levels = [n_start for i in range(int(Omega.shape[0]))]
        bounds = ot.Interval(Omega.shape[0])
        bounds.setUpperBound([Omega[i,1] for i in np.arange(Omega.shape[0])])
        bounds.setLowerBound([Omega[i,0] for i in np.arange(Omega.shape[0])])
        myGrid = ot.Box(levels, bounds)
        nodes = myGrid.generate()
        nodes=np.array(nodes)
        n_int=n_start+1
        h = (Omega[:,1] - Omega[:,0]) /float(n_int)
        h=float(h.prod())

        weights = np.zeros(nodes.shape[0])
        for i in np.arange(1, len(weights) // 2): #to have an integer value for i
            weights[2 * i] = 2.0 * h / 3.0
            weights[2 * i - 1] = 4.0 * h / 3.0
        weights[-2] = 4.0* h / 3.0
        weights[0] = h / 3.0
        weights[-1] = h / 3.0
        

        W = np.zeros((nodes.shape[0],nodes.shape[0]))
        np.fill_diagonal(W, weights)
 

        # non conditioned  co variance matrix construction at nodes
        D_new, ij = l1_cross_distances(nodes)
        # Set up R_new
        r_new = self.corr(self.theta_, D_new)
        R_new = np.eye(nodes.shape[0]) * (1. + self.nugget)
        R_new[ij[:, 0], ij[:, 1]] = r_new
        R_new[ij[:, 1], ij[:, 0]] = r_new
        K=self.sigma2*R_new
        # construction of B = sqrt(W) * C * sqrt(W)
        sqW = np.sqrt(W)
        B1 = np.dot(K, sqW)
        B = np.dot(sqW, B1)

        eig, vect_star = np.linalg.eig(B)
        # reordering for decreasing eigenvalues
        ind = (-eig).argsort()
        eig=eig[ind]
        vect_star=vect_star[:,ind]

        # Truncature
        crit = 1.0 - np.cumsum(eig) / eig.sum()
        M = int(np.argwhere(crit > eps)[-1])+1

        return eig, vect_star, nodes, weights, M
    
    
     
    
    
    
    
    def K_eig_noncond_randuniform(self, Omega,  nb_vect=None, normalize_Omega=False, n_start=100,eps = 1e-6):
        """
        This function computes the eigenvalues and the eigenvectors of B=sqrt(W) * C * sqrt(W) using the Nyström method with uniformaly distributed points over Omega
        C : the non conditioned co variance matrix for the PCA
        Parameters
        ----------
        self :	All parameters are stored in the Gaussian Process model object.
        nb_vect (optional) : int
            maximal nb of eigenvectors, to avoid memory problems (~stochastic dimension > 4) 
        Omega : array_like
            Integration domain (normalized or not) boundaries (n_features,2) 
        normalize_Omega : boolean
                if False : Omega already normalize : no action 
                elif True : normalization of Omega 
        n_start : int 
            nb of integration points
		eps : float
            Tolerance to achieve for the truncature of the KL expansion
        Returns
        -------
        
        eig : eigenvalues
        vect_star : eigenvectors
        nodes : integration nodes  
        weights : integration weights
        M : nb of retained eigenvalues 

        """
        if normalize_Omega : 
            Omega= (Omega - self.X_mean) / self.X_std
            
        list_distrib=list()
        for i in np.arange(Omega.shape[0]): 
            list_distrib.append(ot.Uniform(Omega[i,0], Omega[i,1]))
        distrib_jointe = ot.ComposedDistribution(list_distrib)
        nodes=np.array(distrib_jointe.getSample(n_start))
        
                  
        levels = [1 for i in range(int(Omega.shape[0]))]
        bounds = ot.Interval(Omega.shape[0])
        bounds.setUpperBound([Omega[i,1] for i in np.arange(Omega.shape[0])])
        bounds.setLowerBound([Omega[i,0] for i in np.arange(Omega.shape[0])])
        myGrid = ot.Box(levels, bounds)
        nodes2 = myGrid.generate()
        nodes2=np.array(nodes2)
        nodes=np.concatenate((nodes,nodes2))
        
        h = (Omega[:,1] - Omega[:,0])
        h=float(h.prod())
        w=h/n_start
        weights = w*np.ones(nodes.shape[0])
        W = np.zeros((nodes.shape[0],nodes.shape[0]))
        np.fill_diagonal(W, weights)
 
        # non conditioned  co variance matrix construction at nodes
        D_new, ij = l1_cross_distances(nodes)
        # Set up R_new
        r_new = self.corr(self.theta_, D_new)
        R_new = np.eye(nodes.shape[0]) * (1. + self.nugget)
        R_new[ij[:, 0], ij[:, 1]] = r_new
        R_new[ij[:, 1], ij[:, 0]] = r_new
        K=self.sigma2*R_new
        # construction of B = sqrt(W) * C * sqrt(W)
        sqW = np.sqrt(W)
        B1 = np.dot(K, sqW)
        B = np.dot(sqW, B1)
        
        
        n_comp= min(n_start, 500)
        crit_f=False
        svd = TruncatedSVD(n_components=n_comp, n_iter=10, random_state=1)
        svd.fit(B)
        eig_new = svd.singular_values_
        eig=eig_new
        vect_star = svd.components_.T
        crit_vect=True
        n_comp=n_comp + min(50,n_start/10)
        if n_comp >= n_start : 
            crit_vect = False
        while crit_vect and crit_f == False : 
            vect_star = svd.components_.T
            eig=eig_new
            svd = TruncatedSVD(n_components=n_comp, n_iter=10, random_state=1)
            svd.fit(B)
            eig_new = svd.singular_values_
            crit = 1.0 - eig.sum() / eig_new.sum()
            crit_f = crit < eps
            n_comp=n_comp + 20
            if nb_vect != None : 
                crit_vect = (n_comp <= nb_vect) and (n_comp < n_start)
                
                
                
        # reordering for decreasing eigenvalues
        ind = (-eig).argsort()
        eig=eig[ind]
        vect_star=vect_star[:,ind]
        M = len(eig)
            
        return eig, vect_star, nodes, weights, M    
    


    
    def eval_eigenfunction_noncond(self, weights, nodes, eig, vect, X_new, M):
        """
        This function computes the eigenfunctions phi(x) = (phi_1(x),..,phi_M(x)) from the result
        of the Nystrom method (K_eig_noncond_randuniform or K_eig_noncond_grid)

        Parameters
        ----------
        self :	All parameters are stored in the Gaussian Process model object.
        weigths : array_like
            Integration weights
        nodes : array_like 
            Integration nodes
        eig : array_like
            eigenvalues
        vect : array_like
            eigenvectors
        M : int
            truncature number
        X_new : n_mc x n_features array
            n_mc : nb of evaluation points
            
        Returns
        -------
        phi : array_like M x n_mc
        column k : vector of eigenfunctions phi(x_k) = (phi_1(x_k),..,phi_M(x_k)) for the k th points x_k of the MC 
           
        """
        n_nodes=nodes.shape[0]
        n_mc=X_new.shape[0]
        # computation of the non conditioned co var matrix at integration nodes and new points 
        Dx = manhattan_distances(X_new, Y=nodes, sum_over_features=False)
        r_new = self.corr(self.theta_, Dx)
        del Dx
        del X_new
        del nodes
        K_tot=self.sigma2*r_new.reshape((n_mc,n_nodes))
        del r_new
        W = np.zeros((n_nodes,n_nodes))
        np.fill_diagonal(W, weights)
        phi_int= np.sqrt(W).dot(vect[:,:M])
        del W
        del vect
        # phi_i(x) =sum_j w_j * u_ij * cov(x,x_j)
        if n_nodes < 3000 :
            phi=np.einsum('ki,jk->ij', phi_int, K_tot) # matrix M x n_mc
        # ALTERNATIVE MEMOIRE /!\
        else : 
            if n_mc < 30000 :
                phi=np.einsum('ki,jk->ij', phi_int, K_tot) # matrix M x n_mc
            else :
                phi=np.zeros((M,n_mc))
                for i in np.arange(M) : 
                    phi[i,:] = np.einsum('ki,jk->ij', phi_int[:,i].reshape((n_nodes,1)), K_tot) 
                
        
        return phi
    
        
    def eval_phi_KL(self, weights, nodes, eig, vect, M, x_new):
        """
        evaluation of eigenfunctions phi(x) = (phi_1(x),..,phi_M(x)) 
        """
        if self.normalize == True:
                x_new = (x_new- self.X_mean) / self.X_std
        traj=np.concatenate((self.X,x_new))
        phi=self.eval_eigenfunction_noncond(weights, nodes, eig, vect, traj, M)
        return phi
    
    
    def eval_mean_t(self, x_new,y_traj):
        """
        This function computes mu_t=mu_tilde(x) the mean of the non conditioned GP 

        Parameters
        ----------
        self :	All parameters are stored in the Gaussian Process model object.
        x_new :  n_mc x n_features array
        y_traj :  n_traj x n_doe array
                trajectories at the doe points
                
        Returns
        -------
        
        mu_t :  array_like n_mc x n_traj
   
        """
        n_traj=y_traj.shape[0]

        n_eval=x_new.shape[0]
        # Initialize output
        n_samples, n_features = self.X.shape
        n_doe,n_targets=self.y.shape
        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(x_new, Y=self.X, sum_over_features=False)
        # Get regression function and correlation
        f = self.regr(x_new)
        r = self.corr(self.theta_, dx).reshape(n_eval, n_samples)
         ##########
        mu_t =np.zeros((n_eval,n_traj))
        y_mean=y_traj.mean(axis=1)
        y_std=y_traj.std(axis=1)
        if self.normalize == True:
                y_traj = (y_traj- y_mean.reshape(n_traj,1)) / y_std.reshape(n_traj,1)
        Yt = linalg.solve_triangular(self.C_, y_traj.T, lower=True)
        beta_t = linalg.solve_triangular(self.G_, np.dot(self.Q_.T, Yt))
        rho = Yt - np.dot(self.Ft_, beta_t)
        gamma_t = linalg.solve_triangular(self.C_.T, rho)
        y_ = np.dot(f, beta_t) + np.dot(r, gamma_t)
        mu_t = y_mean + y_std * y_
        
        # mu_t : L : n_mc et C : n_traj 
        return mu_t
    
    
 
    def eval_KL_expansion_trajectories(self, weights, nodes, eig, vect, M, x_new,xi_traj):
        """
        This function computes the KL expansion of the GP 
        
        pre-recquired operations : 
            eigenvalues and the eigenvectors computation using : K_eig_noncond_randuniform or K_eig_noncond_grid

        Parameters
        ----------
        self :	All parameters are stored in the Gaussian Process model object.
        weigths : array_like
            Integration weights
        nodes : array_like 
            Integration nodes
        eig : array_like
            eigenvalues
        vect : array_like
            eigenvectors
        M : int
            trunc number
        x_new : n_mc x n_features array
        xi_traj : array n_traj x (M x (n_doe + n_mc) )
            random vector associated to a traj
            
        Returns
        -------
        
        H_x :  array_like n_traj x n_mc
                trajectories of the GP at points x_new
        phi : array_like M x n_mc
      

        """
        mu=self.predict(x_new)
        if self.normalize == True:
                x_new = (x_new- self.X_mean) / self.X_std
        n_doe, n_features = self.X.shape
        n_targets=self.y.shape[1]
        n_mc,n=x_new.shape
        eig_t=eig[:M].reshape(1,M) 
        # Covariance matrix
        n_traj=xi_traj.shape[1]
        traj=np.concatenate((self.X,x_new))
        phi=self.eval_eigenfunction_noncond(weights, nodes, eig, vect, traj, M)
        H_x=np.zeros((n_traj,n_mc))
        ###
        eig_m=np.zeros((M,M))
        np.fill_diagonal(eig_m,np.sqrt(1./eig_t))
        c=(eig_m.dot(xi_traj[:,:]).T).dot(phi)
        H_tot=c.real
        # H_tot : L : traj et C : doe + mc 
        y_traj=H_tot[:,:n_doe]
        H_0=H_tot[:,n_doe:]
        mu_t=self.eval_mean_t(x_new,y_traj)
        mu_corr=mu.reshape(n_mc, n_targets)-mu_t
        H_x=mu_corr.T+H_0

        return H_x,phi

    
    def eval_KL_expansion_trajectories_phi(self, weights, nodes, eig, vect, M, x_new,xi_traj,phi):
        """
        This function computes the KL expansion of the GP 
        
        pre-recquired operations : 
            eigenvalues and the eigenvectors computation using : K_eig_noncond_randuniform or K_eig_noncond_grid
            evaluation of eigenfunctions using : eval_phi_KL
            
        Parameters
        ----------
        self :	All parameters are stored in the Gaussian Process model object.
        weigths : array_like
            Integration weights
        nodes : array_like 
            Integration nodes
        eig : array_like
            eigenvalues
        vect : array_like
            eigenvectors
        M : int
            trunc number
        x_new : n_mc x n_features array
        xi_traj : array n_traj x (M x (n_doe + n_mc) )
            random vector associated to a traj
        phi : array_like M x n_mc
            eigenfunctions at points x_new
                
        Returns
        -------
        
        H_x :  array_like n_traj x n_mc
                trajectories of the GP at points x_new
        phi : array_like M x n_mc
        """
        mu=self.predict(x_new)
        if self.normalize == True:
                x_new = (x_new- self.X_mean) / self.X_std
        n_doe, n_features = self.X.shape
        n_targets=self.y.shape[1]
        n_mc,n=x_new.shape
        eig_t=eig[:M].reshape(1,M) 
        # Covariance matrix
        n_traj=xi_traj.shape[1]
        H_x=np.zeros((n_traj,n_mc))
        ###
        eig_m=np.zeros((M,M))
        np.fill_diagonal(eig_m,np.sqrt(1./eig_t))
        c=(eig_m.dot(xi_traj[:,:]).T).dot(phi)
        H_tot=c.real
        # H_tot : L : traj et C : doe + mc 
        y_traj=H_tot[:,:n_doe]
        H_0=H_tot[:,n_doe:]
        mu_t=self.eval_mean_t(x_new,y_traj)
        mu_corr=mu.reshape(n_mc, n_targets)-mu_t
        H_x=mu_corr.T+H_0

        return H_x,phi
           
      
   
        

    
    
    
    
    
    def eval_condK(self, weights, nodes, eig, vect, M, x_new):
        """
        This function computes the approximated conditioned covariance matrix at points x_new using the KL representation of the GP 

        Parameters
        ----------
        self :	All parameters are stored in the Gaussian Process model object.
        weigths : array_like
            Integration weights
        nodes : array_like 
            Integration nodes
        eig : array_like
            eigenvalues
        vect : array_like
            eigenvectors
        M : int
            trunc number
        x_new : n_mc x n array
            n : coord nb of points 
        xi_traj : array n_traj x (M x (n_doe + n_mc) )
            random vector associated to a traj
            
        Returns
        -------
        
        K :  array_like n_mc x n_mc
            conditioned covariance matrix 
        """

        C = self.C
        Ft = self.Ft
        n_eval, _ = x_new.shape
        n_samples, n_features = self.X.shape
        if self.normalize == True:
                x_new = (x_new- self.X_mean) / self.X_std
        dx = manhattan_distances(x_new, Y=self.X, sum_over_features=False)
        r=self.corr(self.theta_, dx).reshape(n_eval, n_samples)
        f = self.regr(x_new)
        n_mc,n=x_new.shape
        eig_t=eig[:M].reshape(1,M) 
        # Covariance matrix
        phi=self.eval_eigenfunction_noncond(weights, nodes, eig, vect, x_new, M)
        phi_0=np.sqrt(1./eig_t).reshape(M,1)*phi
        R_new=np.einsum('ki,jk->ij', phi_0, phi_0.T)
        rt = linalg.solve_triangular(C, r.T, lower=True)
        M =self.sigma2* np.dot(rt.T, rt)
        M[M > 1.0] = 1.0
        if self.beta0 is None:
            # Universal Kriging
            T1 = f - np.dot(rt.T, Ft)
            T2 = linalg.inv(np.dot(Ft.T, Ft))
            T3 = np.dot(T2, T1.T)
            M2 = self.sigma2*np.dot(T1, T3)
            K = R_new - M + M2
        else:
            # Ordinary Kriging
            K =R_new - M
        return K
