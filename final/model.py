# MIT License
#
# Copyright (c) 2017 Luca Angioloni
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model
import numpy as np
import keras
from keras import *
from keras.layers import Bidirectional, Dense, Masking, Embedding
from keras.layers import LSTM, TimeDistributed
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, AveragePooling1D, MaxPooling1D, TimeDistributed, LeakyReLU, BatchNormalization
from keras import optimizers, callbacks
from keras.regularizers import l2
# import keras.backend as K
import tensorflow as tf
import time
from keras.callbacks import TensorBoard
import math





LABEL_SET = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
AMINO_ACIDS = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L',
    'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']

WINDOW_SIZE = 21
NUM_AMINO_ACIDS = len(AMINO_ACIDS)
NUM_LABELS = len(LABEL_SET)

SEQUENCE_LIMIT = 700
NUM_ORIG_FEATURES = 57

# Indices of the dataset samples
TRAINING_RANGE = (0, 5600)
TEST_RANGE = (5605, 5877)
VALIDATION_RANGE = (5877, 6133)

INDIV_INPUT_DIM = NUM_AMINO_ACIDS
INPUT_DIM = WINDOW_SIZE * INDIV_INPUT_DIM
HIDDEN_DIM = int(math.sqrt(len(LABEL_SET) * INPUT_DIM))
OUTPUT_DIM = len(LABEL_SET)
import dataset

do_summary = True

LR = 0.0005
drop_out = 0.5
batch_dim = 64
nn_epochs = 25

#loss = 'categorical_hinge' # ok
loss = 'categorical_crossentropy' # best standart
#loss = 'mean_absolute_error' # bad
#loss = 'mean_squared_logarithmic_error' # new best (a little better)


early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='min')

#filepath="NewModel-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath="Whole_CullPDB-best.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


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


def CNN_model():
    # We fix the window size to 11 because the average length of an alpha helix is around eleven residues
    # and that of a beta strand is around six.
    # ref: https://www.researchgate.net/publication/285648102_Protein_Secondary_Structure_Prediction_Using_Deep_Convolutional_Neural_Fields

    m = Sequential()
    parallel_model = m



    # main_input = Masking(mask_value=23)(main_input)
    '''
    m.add(Embedding(5600,output_dim= 21, input_length=700))


    m.add(Bidirectional(LSTM(256, activation='tanh',
                                      return_sequences=True), merge_mode='concat'))
    m.add(Dropout(drop_out))
    m.add(Bidirectional(LSTM(256, return_sequences=True)))
    m.add(Dropout(0.5))

    m.add(TimeDistributed(Dense(256)))
    m.add(Dense(256, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(256, activation='relu'))
    m.add(Dropout(0.5))

    m.add(TimeDistributed(Dense(8, activation='softmax')))  # <----
    # m.add(Conv1D(dataset.num_classes, 11, padding='same'))
    # m.add(TimeDistributed(Activation('softmax')))'''
    parallel_model.add(Masking(mask_value=0,
                           input_shape=(SEQUENCE_LIMIT, 21)))

    parallel_model.add(Bidirectional(LSTM(64, activation='tanh',
                                      return_sequences=True), merge_mode='concat'))
    parallel_model.add(Dropout(0.5))
    parallel_model.add(Bidirectional(LSTM(64, return_sequences=True)))
    parallel_model.add(Dropout(0.5))

    '''m.add(TimeDistributed(Dense(HIDDEN_DIM)))
    m.add(Dropout(0.5))'''
    parallel_model.add(Dense(64,activation='relu'))
    parallel_model.add(Dropout(0.5))
    parallel_model.add(Dense(64, activation='relu'))
    parallel_model.add(Dropout(0.5))

    parallel_model.add(TimeDistributed(Dense(OUTPUT_DIM,activation='softmax')))
    # m.add(Conv1D(dataset.num_classes, 11, padding='same', activation='softmax', input_shape=(dataset.sequence_len, dataset.amino_acid_residues)))
    opt = optimizers.Adam(lr=LR)

    parallel_model.compile(optimizer=opt,
              loss=loss,
              metrics=['accuracy', 'mae'])
    if do_summary:
        print("\nHyper Parameters\n")
        print("Learning Rate: " + str(LR))
        print("Drop out: " + str(drop_out))
        print("Batch dim: " + str(batch_dim))
        print("Number of epochs: " + str(nn_epochs))
        print("\nLoss: " + loss + "\n")

        parallel_model.summary()

    return m


if __name__ == '__main__':
    print("This script contains the model")