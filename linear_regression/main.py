import numpy as np
import helper


class LinReg:
    """
    This class implements the linear regression model
    Fit the model using .fit() method
    Predict the target variable using .predict() method



    After fitting the model, the following attributes are available:
    self.beta: the coefficients of the model
    self.residuals: the residuals of the model
    self.mse: the mean squared error of the model
    self.rmse: the root mean squared error of the model
    self.r2: the r2 score of the model
    self.r2_adj: the adjusted r2 score of the model
    """

    def __init__(self):

        self.bias = None
        self.y = None
        self.x = None
        self.beta = None
        self.residuals = None
        self._mse = None
        self._rmse = None
        self._r2 = None
        self._r2_adj = None
        self.y_hat = None
        self.max_alpha = None
        self.verbose = None
        self.max_iter = None
        self.min_alpha = None
        self.history = {}
        self.model_method = None
        self.alpha = None
        self.diff_w = None
        self.xtx = None
        self.xtx_inv = None
        self.xty = None
        self.intercept = None
        self._y_mean = None
        self._tss = None
        self._aic = None
        self._bic = None
        # Add metaclass to return warning when user tries to access these variables before fitting the model
        return




    def fit(self, x, y, intercept=False, verbose=False, method="MLE", alpha=0.01, max_iter=100):
        """
        :param x: independent variables
        :param y: target variable
        :param method: the method to fit the model, OLS or GD
                       Use GD for large data, function throws an error,
                       if OLS is used on data with size > 50000
        :param intercept: whether the intercept is pre-defined or not
        :param verbose: Boolean, whether to print the summary of the model or not
        :param alpha: learning rate for gradient descent - only applicable when method = GD
        :param max_iter: maximum number of iterations - only applicable when method = GD
        :return:
        """
        # add the intercept term to x
        self.x = x
        self.y = y
        self.verbose = verbose
        self.max_iter = max_iter
        self.alpha = alpha
        self.intercept = intercept
        self.method = method
        self.x = helper.check_if_oned(
            self.x)  # check if the independent variable is one dimensional, if so, reshapes it.
        if self.method == "ols":
            if self.x.size < 50000:
                return self.fit_ols(x=self.x, y=self.y, verbose=self.verbose)
            else:
                raise AssertionError("The data is too large for OLS method. Consider using gradient descent method")
        elif self.method == "gd":
            if self.verbose:
                print("Using gradient descent method to find the solution")
            return self.fit_gd(self.x, self.y, verbose = self.verbose, alpha=self.alpha, max_iter=self.max_iter)
        elif self.method in ["poisson","exponential","gamma"]:
            if self.verbose:
                print("Initializing")
            return self.fit_mle(self.x, self.y, method = self.method, verbose= self.verbose)

    def fit_ols(self, x, y, verbose=False, intercept=False):
        self.x = x
        self.y = y
        self.verbose = verbose
        self.x = helper.check_if_oned(self.x)
        if not self.intercept:
            self.x = helper.add_intercept(self.x)
        if size(self.x) > 50000:
            print("The data is too large for OLS method. Consider using gradient descent method")
            self.fit_gd(self.x, self.y, verbose=True)
            return self
        # print("Using OLS method to find the solution")
        self.model_method = "OLS"
        # using Moore-Penrose pseudo-inverse to calculate the inverse of xtx
        u, s, v= np.linalg.svd(np.dot(self.x.T, self.x))
        s = np.diag(s)
        x_sq_reg_inv = V.dot(np.linalg.pinv(s)).dot(U.T)
        self.beta = x_sq_reg_inv.dot(self.x.T).dot(self.y)

        self._cal_metrics()
        self._verbose_print()
        return self

    def fit_mle(self, x, y, verbose=False, intercept=False, method="poisson"):
        if method == "gamma":
            d_cost_function =
            return d_cost_function
        elif method == "poisson":
            d_cost_function =
            return d_cost_function
        elif method == "exponential":
            d_cost_function =
            return d_cost_function
        self.x = x
        self.y = y
        self.verbose = verbose
        self.x = helper.check_if_oned(self.x)




    def fit_gd(self, x, y, verbose=False, alpha=0.001, max_iter=100, intercept=False, max_alpha=None,
               min_alpha=None):
        """
        calculates the coefficients using gradient descent method.

        :param x:
        :param y:
        :param verbose:
        :param alpha:
        :param max_iter:
        :param intercept:
        :param max_alpha:
        :param min_alpha:
        :return:
        """
        # re-initialising self.x in case user uses fit_gd() directly
        self.x = x
        self.y = y
        self.verbose = verbose
        self.alpha = alpha
        self.max_iter = max_iter
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.history = {}
        self.model_method = "GD"
        self.x = helper.check_if_oned(self.x)
        self.x = helper.normalise(self.x)
        self.y = helper.check_if_oned(self.y)
        if not intercept:
            self.x = helper.add_intercept(self.x)
        if self.verbose:
            print("Using gradient descent method to find the solution")

        self.beta = helper.initialize_weights(self.x.shape[1])

        for i in range(self.max_iter):
            """if self.max_alpha is not None:
                if self.max_iter > 0.8 * self.max_iter:
                    self.alpha = self.min_alpha
                else:
                    self.alpha = self.max_alpha"""

            self.y_hat = np.dot(self.x, self.beta)
            self.residuals = self.y_hat - self.y
            # print(f"residuals: {self.residuals}")
            self.diff_w = -(self.y - self.y_hat).dot(self.x)
            self.beta = self.beta - (self.alpha * self.diff_w)
            if i % 100 == 0:
                print(self.beta)
            # bias is also updated. The intercept allows automatic calculation
            # self.history.update(self._cal_metrics())

        self._verbose_print()
        return self

    def _verbose_print(self):
        if self.verbose:
            self.summary()
            print(f"The model has been fitted successfully with r squared: {self._r2}")

    @staticmethod
    def predict(self, x):
        # sample prediction
        pred = np.dot(self.x, self.beta)
        return pred

    def summary(self):
        print(f"The model has been fitted successfully with {self.model_method}")
        print("The coefficients are: ", self.beta)
        # print("The residuals are: ", self.residuals)
        print("The mean squared error is: ", self._mse)
        print("The root mean squared error is: ", self._rmse)
        print("The r2 score is: ", self._r2)
        print("The adjusted r2 score is: ", self._r2_adj)
        return

    def _cal_metrics(self):
        # Calculate metrics like MSE, RMSE, R-squared, adjusted R-squared, TSS, RSS, etc.

        if self.y_hat is None:
            self.y_hat = np.dot(self.x, self.beta)
        if self.residuals is None:
            self.residuals = self.y - self.y_hat
        self._calculate_mse()
        self._calculate_rmse()
        self._calculate_tss()
        self._calculate_r2()
        self._calculate_r2_adj()
        self._calculate_aic()
        self._calculate_bic()

        return {'mse': self._mse, 'rmse': self._rmse}

    def _calculate_mse(self):
        if self._mse is None:
            self._mse = np.mean(self.residuals ** 2)
            return self._mse
        return self._mse

    def _calculate_rmse(self):
        if self._rmse is None:
            self._rmse = np.sqrt(self._mse)
            return self._rmse
        return self._rmse

    def _calculate_r2(self):
        if self._r2 is None:
            self._r2 = 1 - (self._mse / self._tss)
            return self._r2
        return self._r2

    def _calculate_tss(self):
        if self._tss is None:
            self._tss = np.sum((self.y - np.mean(self.y)) ** 2)
            print(f"tss: {self._tss}")
        return self._tss

    def _calculate_r2_adj(self):
        if self._r2_adj is None:
            self._r2_adj = 1 - (1 - self._r2) * (len(self.y) - 1) / (len(self.y) - self.x.shape[1] - 1)
        return self._r2_adj

    def _calculate_aic(self):
        if self._aic is None:
            self._aic = helper.calculate_aic(len(self.x), self._mse, self.x.shape[1])
        return self._aic

    def _calculate_bic(self):
        if self._bic is None:
            self._bic = helper.calculate_bic(len(self.x), self._mse, self.x.shape[1])
        return self._bic

    def get_history(self):
        if self.history == {}:
            raise AssertionError("The model has not been fitted yet or the method is not GD")
        return self.history
