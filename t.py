
import numpy as np
import random as rn
import os

# set seed
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import logging
logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# np.save('x_train',x_train)
# np.save('y_train',y_train)
# np.save('x_test',x_test)
# np.save('y_test',y_test)
# y_train = np.load('y_train.npy')
# x_train = np.load('x_train.npy')
# x_test = np.load('x_test.npy')
# y_test = np.load('y_test.npy')


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def split_client_dataset(num_clients: int, len_dataset: int, fixed_size: int=None):
    """Generate a index list based on number of clients and length of dataset.
      Args:
          num_clients: number of clients
          len_dataset: Number of samples in the whole dataset
          fixed_size: If setted, each client will only take a fixed number of samples.

      Returns:
          A nested list with index list for each client.
    """
    ind_list = np.linspace(0, len_dataset - 1, len_dataset).astype(np.int32)
    client_data_list = []
    size = int(len_dataset / num_clients)
    for client in range(num_clients):
        if fixed_size is not None:
            size = fixed_size
        data_list = np.random.choice(ind_list, size)

        ind_list = [i for i in ind_list if i not in data_list]
        client_data_list.append(data_list)
    return client_data_list


def get_model(num_classes):
    """Get a MLP model in keras."""

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(784, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def init(model):
    """In keras if you don't run a funcition of the model, the model's wight would be empty [0].
       This is only for weight initilization.
    """
    model.evaluate(x_test[0:1,...], y_test[0:1,...], verbose=0)

t_round = 8
num_clients = 10
len_dataset = x_train.shape[0]
global_model = get_model(10)
init(global_model)
acc_list = []

for r in range(t_round):
    print("Round: "+str(r+1)+" started.")

    # Size of weight based on the model
    weight_acc = np.asarray([np.zeros((784, 784)), np.zeros(
        (784,)), np.zeros((784, 10)), np.zeros((10,))])

    # Generate index lists.
    client_data_list = split_client_dataset(num_clients, len_dataset, fixed_size=60)

    for c in range(num_clients):
        model = get_model(10)
        init(model)
        model.set_weights(global_model.get_weights())
        param_before = np.asarray(model.get_weights())
        # if param_before.shape[0] == 0:
        #     param_before = np.asarray([np.zeros((784, 784)), np.zeros(
        #         (784,)), np.zeros((784, 10)), np.zeros((10,))])

        # Get index list
        ind = client_data_list[c]
        c_feature = np.take(x_train, ind, axis=0)
        c_label = np.take(y_train, ind, axis=0)

        # Train client
        model.fit(c_feature, c_label,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  validation_data=(c_feature, c_label))
        param_after = np.asarray(model.get_weights())

        # Average gradient by percentage. If number of samples weren't the same.
        # Change this part according to FedAvg.
        weight_acc += (param_after) * (1 / num_clients)
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Client'+str(c+1)+' with accuracy:', score[1])

    global_model.set_weights(weight_acc)
    score = global_model.evaluate(x_test, y_test, verbose=0)
    print('Global test loss:', score[0])
    print('Global accuracy:', score[1])
    acc_list.append(score[1])
print(acc_list)


import matplotlib.pyplot as plt
plt.plot(acc_list)
plt.show()
