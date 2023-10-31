import numpy as np
from helper import add_intercept


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

        self.y = None
        self.x = None
        self.beta = None
        self.residuals = None
        self.mse = None
        self.rmse = None
        self.r2 = None
        self.r2_adj = None
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
        # Add metaclass to return warning when user tries to access these variables before fitting the model
        return

    def fit(self, x, y, intercept="False", verbose='False', method="OLS", alpha=0.01, max_iter=100, ):
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
        if method == "OLS":
            if self.x.size < 50000:
                if self.verbose:
                    print("Using OLS method to find the solution")
                return self.fit_ols(self, self.x, self.y)
            else:
                raise AssertionError("The data is too large for OLS method. Consider using gradient descent method")
        elif method == "GD":
            if self.verbose:
                print("Using gradient descent method to find the solution")
            return self.fit_gd(self, self.x, self.y)

    def fit_gd(self, x, y, verbose='False', alpha=0.01, max_iter=100, intercept="False", max_alpha=None,
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

        self.alpha = alpha
        self.max_iter = max_iter
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        # re-initialising self.x in case user uses fit_gd() directly
        self.x = x
        if not intercept:
            self.x = add_intercept(self.x)
        if verbose:
            print("Using gradient descent method to find the solution")
        self.beta = np.zeros(x.shape[1])
        self.history = {}
        self.model_method = "GD"
        for i in range(self.max_iter):
            if self.max_alpha is not None:
                if self.max_iter > 0.8 * self.max_iter:
                    self.alpha = self.min_alpha
                else:
                    self.alpha = self.max_alpha
            self.y_hat = np.dot(self.x, self.beta)
            self.residuals = self.y - self.y_hat
            self.diff_w = (-1 / len(x)) * np.dot(self.x.T, self.residuals)
            self.beta = self.beta - self.alpha * self.diff_w
            # bias is also updated. The intercept allows automatic calculation
            self.history.update(self.cal_metrics())
        if verbose:
            self.summary()
            return print("The model has been fitted successfully")
        return

    def fit_ols(self, x, y, verbose='False'):
        print("Using OLS method to find the solution")
        self.model_method = "OLS"
        # calculating the beta
        xtx = np.dot(self.x.T, self.x)
        xtx_inv = np.linalg.inv(xtx)
        xty = np.dot(self.x.T, self.y)
        self.beta = np.dot(xtx_inv, xty)
        if verbose:
            self.summary()
            return print("The model has been fitted successfully")
        return

    def predict(self, x):
        # sample prediction
        pred = np.dot(self.x, self.beta)
        return pred

    def summary(self):
        print(f"The model has been fitted successfully with {self.model_method}")
        print("The coefficients are: ", self.beta)
        print("The residuals are: ", self.residuals)
        print("The mean squared error is: ", self.mse)
        print("The root mean squared error is: ", self.rmse)
        print("The r2 score is: ", self.r2)
        print("The adjusted r2 score is: ", self.r2_adj)
        return

    def cal_metrics(self):
        self.y_hat = np.dot(self.x, self.beta)
        # calculating the residuals
        self.residuals = self.y - self.y_hat
        # calculating the mean squared error
        self.mse = np.mean(self.residuals ** 2)
        # calculating the root mean squared error
        self.rmse = np.sqrt(self.mse)
        # calculating the r2 score
        self.r2 = 1 - (np.var(self.residuals) / np.var(self.y))
        # calculating the adjusted r2 score
        self.r2_adj = 1 - (1 - self.r2) * (len(self.y) - 1) / (len(self.y) - self.x.shape[1] - 1)

        return {'residuals': self.residuals, 'mse': self.mse, 'rmse': self.rmse, 'r2': self.r2, 'r2_adj': self.r2_adj}

    def get_history(self):
        if self.history == {}:
            raise AssertionError("The model has not been fitted yet or the method is not GD")
        return self.history
