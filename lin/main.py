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
        self.y_mean = None
        # Add metaclass to return warning when user tries to access these variables before fitting the model
        return

    def fit(self, x, y, intercept=False, verbose=False, method="OLS", alpha=0.01, max_iter=100):
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
        if self.method == "OLS":
            if self.x.size < 50000:
                return self.fit_OLS(x=self.x, y=self.y, verbose=self.verbose)
            else:
                raise AssertionError("The data is too large for OLS method. Consider using gradient descent method")
        elif self.method == "GD":
            if self.verbose:
                print("Using gradient descent method to find the solution")
            return self.fit_GD(self.x, self.y)
        elif self.method == "MLE":
            if self.verbose:
                print("Initializing")
            return self.fit_MLE(self.x, self.y)

    def fit_GD(self, x, y, verbose=False, alpha=0.01, max_iter=100, intercept=False, max_alpha=None,
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
        self.verbose = verbose
        self.alpha = alpha
        self.max_iter = max_iter
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.history = {}
        self.model_method = "GD"
        self.x = helper.check_if_oned(self.x)
        if not intercept:
            self.x = helper.add_intercept(self.x)
        if self.verbose:
            print("Using gradient descent method to find the solution")
        
        self.beta = np.ones((1,self.x.shape[1]))
        
        for i in range(self.max_iter):

            """if self.max_alpha is not None:
                if self.max_iter > 0.8 * self.max_iter:
                    self.alpha = self.min_alpha
                else:
                    self.alpha = self.max_alpha"""
            
            self.y_hat = np.dot(self.x, self.beta.T)
            print(self.y_hat.shape)
            self.residuals = self.y - self.y_hat
            self.diff_w = (-1 / len(x)) * np.dot(self.x.T, self.residuals)
            self.beta = self.beta - self.alpha * self.diff_w
            # bias is also updated. The intercept allows automatic calculation
            self.history.update(self.cal_metrics())

        if self.verbose:
            self.summary()
            return print(f"The model has been fitted successfully with r squared: {self.r2}")
        return self.r2_adj

    def fit_MLE(self, x, y, verbose=False, intercept=False):
        self.x = x
        self.y = y
        self.verbose = verbose
        self.x = helper.check_if_oned(self.x)
        print("Using MLE method to find the solution")
        self.model_method = "MLE"
        # calculating the beta
        # Calculate the mean of X and y
        if np.isnan(self.x).any():
            raise ValueError("X contains NaN values. Please check your data")
        self.x_mean = np.mean(self.x)
        self.y_mean = np.mean(self.y)
        print(self.x_mean, self.y_mean)
        # Calculate the slope (coefficient)
        numerator = np.sum((self.x - self.x_mean) * (self.y - self.y_mean))
        denominator = np.sum((self.x - self.x_mean) ** 2)
        # if denominator == 0:
        #    raise ValueError("Denominator is zero. Check your data.")
        self.slope = (numerator / (denominator))
        print(self.slope)
        # Calculate the intercept
        self.bias = self.y_mean - (self.slope * self.x_mean)
        self.beta = [self.bias].extend(self.slope)
        self.cal_metrics()
        if self.verbose:
            self.summary()
            return print(f"The model has been fitted successfully with r squared: {self.r2}")
        return self.r2_adj

    def fit_OLS(self, x, y, verbose=False, intercept=False):
        self.x = x
        self.y = y
        self.verbose = verbose
        self.x = helper.check_if_oned(self.x)
        if not self.intercept:
            self.x = helper.add_intercept(self.x)
        print("Using OLS method to find the solution")
        self.model_method = "OLS"
        # calculating the beta
        xtx = np.dot(self.x.T, self.x)
        xtx_inv = np.linalg.inv(xtx)
        xty = np.dot(self.x.T, self.y)
        self.beta = np.dot(xtx_inv, xty)
        self.cal_metrics()
        if self.verbose:
            self.summary()
            return print(f"The model has been fitted successfully with r squared: {self.r2}")
        return self.r2_adj

    def predict(self, x):
        # sample prediction
        pred = np.dot(self.x, self.beta)
        return pred

    def summary(self):
        print(f"The model has been fitted successfully with {self.model_method}")
        print("The coefficients are: ", self.beta)
        # print("The residuals are: ", self.residuals)
        print("The mean squared error is: ", self.mse)
        print("The root mean squared error is: ", self.rmse)
        print("The r2 score is: ", self.r2)
        print("The adjusted r2 score is: ", self.r2_adj)
        return

    @property
    def cal_metrics(self):
        #should I have a different class for metrics
        self.y_hat = np.dot(self.x, self.beta)
        # calculating the residuals
        self.residuals = self.y - self.y_hat
        # calculating the mean squared error
        self._mse = np.mean(self.residuals ** 2)
        # calculating the root mean squared error
        self._rmse = np.sqrt(self.mse)

        # calculating the r2 score
        if self.y_mean is None:
            self.y_mean = np.mean(self.y)
        self.y_pred_mean = np.mean(self.y_hat)
        self.tss = np.sum((self.y - self.y_mean) ** 2)
        self.rss = np.sum((self.y - self.y_hat) ** 2)
        self.r2 = 1 - (self.rss / self.tss)
        # calculating the adjusted r2 score
        self.r2_adj = 1 - (1 - self.r2) * (len(self.y) - 1) / (len(self.y) - self.x.shape[1] - 1)

        # Additional Scores

        self.aic = helper.calculate_aic(len(self.x),self.mse, self.x.shape[1])
        self.bic = helper.calculate_bic(len(self.x),self.mse, self.x.shape[1])
        return {'residuals': self.residuals, 'mse': self.mse, 'rmse': self.rmse, 'r2': self.r2, 'r2_adj': self.r2_adj}


    def get_history(self):
        if self.history == {}:
            raise AssertionError("The model has not been fitted yet or the method is not GD")
        return self.history
