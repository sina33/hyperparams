from keras import activations
from keras import optimizers
from matplotlib.pyplot import hist
import random
import numpy as np


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
    p = get_rand_int(low, high)
    return 10 * random.random() * (10 ** p)


def get_optimizer(exclude=None):
    selection = [optimizers.SGD, optimizers.RMSprop, optimizers.Adagrad, optimizers.Adadelta,
                optimizers.Adam, optimizers.Adamax, optimizers.Nadam]
    if exclude is not None:
        selection.remove(exclude)
    return random.choice(selection)


def get_activation(exclude=None):
    selection = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    if exclude is not None:
        selection.remove(exclude)
    return random.choice(selection)
    

KEYS = ['filters', 'kernel_size', 'activation', 'pool_size', 'units', 'rate', 'optimizer', 'lr']

def get_param(param):
    pass

x = [get_optimizer() for i in range(10)]
print(x)