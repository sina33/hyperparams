# Lenet-5
# Build a deep convolutional neural network to classify MNIST digits

# set seed for reproducibility
import numpy as np
np.random.seed(42)

# load dependencies
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten, MaxPooling2D, Conv2D
from keras.callbacks import EarlyStopping


# load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#print(X_train.shape)
#print(X_test.shape)

# preprocess data
X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32')
X_train /= 255
X_test /= 255
#print(X_train[0])
n_classes = 10
Y_train = keras.utils.to_categorical(Y_train, n_classes)
Y_test = keras.utils.to_categorical(Y_test, n_classes)
#print(Y_train[0])

# def run(conv1=32, conv2=64, kernel=3, pool=2, do1=0.25, dense1=128, do2=0.5, lr=0.01):
def run(params):
    k1 = params['L1']['kernel_size']
    k2 = params['L2']['kernel_size']
    p = params['L3']['pool_size']
    # design neural network architecture
    model = Sequential()
    model.add(Conv2D(params['L1']['filters'], kernel_size=(k1, k1), activation=params['L1']['activation'], input_shape=(28, 28, 1)))
    model.add(Conv2D(params['L2']['filters'], kernel_size=(k2, k2), activation=params['L1']['activation']))
    model.add(MaxPooling2D(pool_size=(p, p)))
    model.add(Dropout(params['L4']['rate']))
    model.add(Flatten())
    model.add(Dense(params['L5']['units'], activation=params['L5']['activation']))
    model.add(Dropout(params['L6']['rate']))
    model.add(Dense(n_classes, activation=params['L7']['activation']))
    # model.summary()

    # configure model
    # opt = keras.optimizers.SGD(lr=lr)
    opt = params['opt'](lr=params['lr'])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # train
    hist = model.fit(X_train,Y_train, batch_size=128, epochs=1, verbose=1, validation_data=(X_test, Y_test), 
                    callbacks=[EarlyStopping(patience=1, min_delta=0.02)])
    return hist.history