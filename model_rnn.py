
import functools
import numpy as np
import  keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Flatten, MaxPooling1D,Conv1D, AveragePooling1D, MaxPooling1D, TimeDistributed, LeakyReLU, BatchNormalization,ZeroPadding1D
from keras import optimizers, callbacks
from keras.regularizers import l2
# import keras.backend as K
import tensorflow as tf
from keras import initializers

import data_2

do_summary = True

LR = 0.0002
drop_out = 0.25
batch_dim = 32
nn_epochs = 60

#loss = 'categorical_hinge' # ok
loss = 'categorical_crossentropy' # best standart
#loss = 'mean_absolute_error' # bad
#loss = 'mean_squared_logarithmic_error' # new best (a little better)


early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='min')

#filepath="NewModel-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath="Whole_CullPDB-best.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

#top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)

#top3_acc.__name__ = 'top3_acc'
def Q8_accuracy(real, pred):
    total = real.shape[0] * real.shape[1]
    correct = 0
    for i in range(real.shape[0]):  # per element in the batch
        for j in range(real.shape[1]): # per aminoacid residue
            if np.sum(real[i, j, :]) == 0:  #  real[i, j, dataset.num_classes - 1] > 0 # if it is padding
                total = total - 1
            else:
                if real[i, j, np.argmax(pred[i, j, :])] > 0:
                    correct = correct + 1

    return correct / total
sequence_len = 700
total_features = 57
amino_acid_residues = 21
num_classes = 8
cnn_width = 17
kt=keras.initializers.he_uniform(seed=None)
def CNN_model():
    m = Sequential()
    m.add(Conv1D(512, 11, padding='same',
                 input_shape=(data_2.sequence_len, data_2.amino_acid_residues)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(Dropout(drop_out))  # <----

    m.add(Conv1D(512, 11, padding='same'))  # <----
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(Dropout(drop_out))  # <----

    m.add(Conv1D(512, 11, padding='same'))  # <----
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(Dropout(drop_out))  #

    m.add(Conv1D(256, 11, padding='same'))  # <----
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(Dropout(drop_out))

    m.add(Conv1D(256, 11, padding='same'))  # <----
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(Dropout(drop_out))

    m.add(Conv1D(256, 11, padding='same'))  # <----
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(Dropout(drop_out))




    m.add(Conv1D(256, 11, padding='same', activation='relu'))  # <----
    m.add(Dense(256,activation="relu"))
    m.add(Dropout(drop_out))
    m.add(Dense(num_classes, activation="softmax"))

    opt = optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    m.compile(optimizer=opt,
              loss=loss,
              metrics=['accuracy', 'mae'])
    if do_summary:
        print("\nHyper Parameters\n")
        print("Learning Rate: " + str(LR))
        print("Drop out: " + str(drop_out))
        print("Batch dim: " + str(batch_dim))
        print("Number of epochs: " + str(nn_epochs))
        print("\nLoss: " + loss + "\n")

        m.summary()

    return m


if __name__ == '__main__':
    print("This script contains the model")