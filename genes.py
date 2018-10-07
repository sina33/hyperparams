from keras import activations
from keras import optimizers
from matplotlib.pyplot import hist
import random
import numpy as np
from math import log10, floor


def get_rand_int(min, max, exclude=None):
    selection = [i for i in range(min, max+1)]
    if exclude is not None:
        selection.remove(exclude) 
    return random.choice(selection)


def get_filters(min=10, max=100, exclude=None):
    return get_rand_int(min, max, exclude)


def get_kernel_size(min=2, max=6, exclude=None):
    return get_rand_int(min, max, exclude)


def get_pool_size(min=2, max=6, exclude=None):
    return get_rand_int(min, max, exclude)


def get_rand_uniform(min, max):
    return round(random.uniform(min, max), 2)


def get_units(min=16, max=256, exclude=None):
    return get_rand_int(min, max, exclude)


def get_dropout_rate(min=0.1, max=0.9):
    return get_rand_uniform(min, max)


def get_learning_rate(low=-4, high=-1):
    def round_to_1(x):
        return round(x, -int(floor(log10(abs(x)))))
    p = get_rand_int(low, high)
    d = 10 * random.random() * (10 ** p)
    return round_to_1(d)


def get_optimizer(exclude=None):
    selection = [optimizers.SGD, optimizers.RMSprop, optimizers.Adagrad, optimizers.Adadelta,
                optimizers.Adam, optimizers.Adamax, optimizers.Nadam]
    if exclude is not None:
        selection.remove(exclude)
    return random.choice(selection)


def get_activation(exclude=None):
    selection = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear', None]
    if exclude is not None:
        selection.remove(exclude)
    return random.choice(selection)
    

KEYS = ['filters', 'kernel_size', 'activation', 'pool_size', 'units', 'rate', 'optimizer', 'lr']
