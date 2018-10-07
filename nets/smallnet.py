# Build a shallow neural network to classify MNIST digits

# set seed for reproducibility
import numpy as np
np.random.seed(42)

# load dependencies
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

# load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#print(X_train.shape)
#print(X_test.shape)

# preprocess data
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')
X_train /= 255
X_test /= 255
#print(X_train[0])
n_classes = 10
Y_train = keras.utils.to_categorical(Y_train, n_classes)
Y_test = keras.utils.to_categorical(Y_test, n_classes)
#print(Y_train[0])

# def run(n1=64, n2=64, ac1='relu', ac2='relu', ini1='ones', ini2='zeros', lr=0.1):
def run(params):
    n1 = params['L1']['units']
    ac1 = params['L1']['activation']
    n2 = params['L2']['units']
    ac2 = params['L2']['activation']
    ac3 = params['L3']['activation']
    opt = params['opt']
    lr = params['lr']
    # design neural network architecture
    model = Sequential()
    model.add(Dense(n1, activation=ac1, input_shape=(784,)))
    model.add(Dense(n2, activation=ac2))
    model.add(Dense(10, activation=ac3))
    # model.summary()

    # configure model
    opt = opt(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr), metrics=['accuracy'])

    # train
    hist = model.fit(X_train, Y_train, batch_size=128, epochs=2, verbose=1, validation_data=(X_test, Y_test))
                    # callbacks=[EarlyStopping(min_delta=0.02)])
    # weights = model.get_weights()
    # for i in range(np.shape(weights)[0]):
    #     print(np.shape(weights[i]))
    return hist.history

# if __name__=="__main__":
#     hist = run(64, 64, 'relu', 'tanh')
#     print(type(hist))
#     for k, v in hist.items():
#         print("{}: {}".format(k, v))