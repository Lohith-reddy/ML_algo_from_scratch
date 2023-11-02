import numpy as np


def initialize_weights(n_features):
    """ Initialize weights randomly [-1/N, 1/N] """

    limit = 1 / math.sqrt(n_features)
    beta = np.random.uniform(-limit, limit, (n_features,))
    return beta

def normalise(x):
    for i in range(x.shape[1]):
        print(f"normalising {i}th column")
        x.iloc[:, i] = (x.iloc[:, i] - np.mean(x.iloc[:, i])) / np.std(x.iloc[:, i])
    return x

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
    aic = (n * np.log(mse) + 2 * num_params)
    return aic

def calculate_bic(n, mse, num_params):
	bic = n * np.log(mse) + num_params * np.log(n)
	return bic