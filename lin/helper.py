import numpy as np


def add_intercept(x):
    intercept = np.ones((x.shape[0], 1))
    return np.concatenate((intercept, x), axis=1)
