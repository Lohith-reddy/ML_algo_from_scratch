import numpy as np


def add_intercept(x):
    intercept = np.ones((x.shape[0], 1))
    x = np.concatenate((intercept, x), axis=1)
    return x

def check_if_oned(x):
    if len(x.shape)==1:
        x = x.to_frame()
        return x
    elif len(x.shape)==0:
        raise AssertionError("The independent variable should have at least one column")
    else:
        return x
    
def calculate_aic(n, mse, num_params):
    aic = (n * log(mse) + 2 * num_params)
    return aic

def calculate_bic(n, mse, num_params):
	bic = n * log(mse) + num_params * log(n)
	return bic