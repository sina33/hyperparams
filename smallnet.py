# Build a shallow neural network to classify MNIST digits

# set seed for reproducibility
import numpy as np
np.random.seed(42)

# load dependencies
import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.metrics import mean_absolute_error

dataset = 'cifar10'

if dataset == 'mnist':
    # load data
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    in_shape = (784,) # reshape to (784,) for Dense or (28, 28, 1) for Conv2D
    # preprocess data
    X_train = X_train.reshape(60000, 784).astype('float32')
    X_test = X_test.reshape(10000, 784).astype('float32')
    X_train /= 255
    X_test /= 255
    n_classes = 10
    Y_train = keras.utils.to_categorical(Y_train, n_classes)
    Y_test = keras.utils.to_categorical(Y_test, n_classes)

elif dataset == 'cifar10':
    # load data
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    in_shape = (3072,) # reshape to (3072,) for Dense or (32, 32, 3) for Conv2D
    # preprocess data
    X_train = X_train.reshape(50000, 3072).astype('float32')
    X_test = X_test.reshape(10000, 3072).astype('float32')
    X_train /= 255.0
    X_test /= 255.0
    n_classes = 10
    Y_train = keras.utils.to_categorical(Y_train, n_classes)
    Y_test = keras.utils.to_categorical(Y_test, n_classes)

# def run(n1=64, n2=64, ac1='relu', ac2='relu', ini1='ones', ini2='zeros', lr=0.1):
def run(params, confusion=False):
    n1 = 64 # params['L1']['units']
    ac1 = 'relu' # params['L1']['activation']
    n2 = 64 # params['L2']['units']
    ac2 = 'relu' # params['L2']['activation']
    ac3 = 'sigmoid' # params['L3']['activation']
    opt = keras.optimizers.SGD # params['opt']
    lr = 0.01 # params['lr']
    # design neural network architecture
    model = Sequential()
    model.add(Dense(n1, activation=ac1, input_shape=in_shape))
    model.add(Dense(n2, activation=ac2))
    model.add(Dense(10, activation=ac3))
    # model.summary()

    # configure model
    opt = opt(lr=lr)
    # loss: mean_absolute_percentage_error
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr), metrics=['accuracy'])

    # train
    callbacks=[EarlyStopping(monitor='val_loss', patience=2)]
    hist = model.fit(X_train, Y_train, batch_size=128, epochs=2, verbose=1, validation_data=(X_test, Y_test))

    if confusion:
        return model, X_test, Y_test
    else:
        return hist.history

# if __name__=="__main__":
#     hist = run(64, 64, 'relu', 'tanh')
#     print(type(hist))
#     for k, v in hist.items():
#         print("{}: {}".format(k, v))