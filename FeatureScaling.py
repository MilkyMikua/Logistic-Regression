# This pack built to rescale features
# Prevent from features's value varies too large result converage issue

import numpy as np

def divideByMax(x):
    '''
    :param x: training examples
    :return: a new set of x_train that divided by the larget number
    '''
    maximum = np.max(x)
    x = x / maximum
    return x


def Zscore (x):
    average = np.mean(x, axis = 0)
    variance = np.var(x, axis=0)
    x = (x - average) / variance

    return x

def meanNormal(x):
    mean = np.mean(x, axis=0)
    diff = np.max(x) - np.min(x)
    x = (x - mean) / diff

    return x

