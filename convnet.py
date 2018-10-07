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

def run(nc1=32, k1=3, nc2=64, k2=3, p=2, do1=0.25, nd1=128, do2=0.5, lr=0.01):
    # design neural network architecture
    model = Sequential()
    model.add(Conv2D(nc1, kernel_size=(k1, k1), activation='relu', input_shape=(28, 28, 1)))
    # model.add(Conv2D(nc2, kernel_size=(k2, k2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(p, p)))
    model.add(Dropout(do1))
    model.add(Flatten())
    model.add(Dense(nd1, activation='relu'))
    model.add(Dropout(do2))
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()

    # configure model
    opt = keras.optimizers.SGD(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # train
    hist = model.fit(X_train,Y_train, batch_size=128, epochs=6, verbose=1, validation_data=(X_test, Y_test), 
                    callbacks=[EarlyStopping(patience=1, min_delta=0)])
    return hist.history