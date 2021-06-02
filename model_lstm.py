import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model
import numpy as np
import keras
from keras import initializers
from keras import *
from keras.layers import Bidirectional, Dense, Masking, Embedding
from keras.layers import LSTM, TimeDistributed, GRU
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, AveragePooling1D, MaxPooling1D, TimeDistributed, LeakyReLU, BatchNormalization,Concatenate
from keras import optimizers, callbacks
from keras.regularizers import l2
# import keras.backend as K
import tensorflow as tf
import time
from keras.callbacks import TensorBoard
import math

from keras_self_attention import SeqSelfAttention



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

LR = 0.002
drop_out = 0.25
batch_dim = 32
nn_epochs = 40

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
    #kt=keras.initializers.he_uniform(seed=None)
    # kt=keras.initializers.he_uniform(seed=None)
    parallel_model = Sequential()
    parallel_model.add(Masking(mask_value=0, input_shape=(SEQUENCE_LIMIT, 21)))
    #parallel_model.add(Embedding(input_dim=(SEQUENCE_LIMIT, 21), output_dim=OUTPUT_DIM, embeddings_initializer='uniform'))
    # parallel_model.add(Flatten())

    parallel_model.add(Bidirectional(LSTM(64, activation='tanh',return_sequences=True), merge_mode='concat'))
    parallel_model.add(Dropout(0.25))
    parallel_model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True), merge_mode='concat'))
    #parallel_model.add(LSTM(256, activation='relu', return_sequences=True))

    parallel_model.add(Dropout(0.25))
    #parallel_model.add(GRU(256, activation='relu', return_sequences=True))
    #parallel_model.add(Dropout(0.25))
    #parallel_model.add(Bidirectional(LSTM(256, activation='tanh', return_sequences=True), merge_mode='concat'))
    # parallel_model.add(LSTM(256, activation='relu', return_sequences=True))

    #parallel_model.add(Dropout(0.25))


    # parallel_model.add(Flatten())
    '''parallel_model.add(SeqSelfAttention(attention_width=15,
    attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
    attention_activation=None,
    kernel_regularizer=keras.regularizers.l2(1e-6),
    use_attention_bias=False,
    name='Attention',))
    parallel_model.add(Dropout(0.25))'''

    parallel_model.add(Dense(512, activation='relu'))
    parallel_model.add(Dropout(0.25))

    parallel_model.add(Dense(512, activation='relu'))
    parallel_model.add(Dropout(0.25))

    parallel_model.add(Dense(512, activation='relu'))
    parallel_model.add(Dropout(0.25))

    parallel_model.add(Dense(512, activation='relu'))
    parallel_model.add(Dropout(0.25))

    parallel_model.add(Dense(512, activation='relu'))
    parallel_model.add(Dropout(0.25))

    parallel_model.add(Dense(256, activation='relu'))
    parallel_model.add(Dropout(.25))

    parallel_model.add(TimeDistributed(Dense(OUTPUT_DIM,activation='softmax')))
    opt = optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

    parallel_model.compile(optimizer=opt, loss=loss, metrics=['accuracy', 'mae'])
    if do_summary:
        print("\nHyper Parameters\n")
        print("Learning Rate: " + str(LR))
        print("Drop out: " + str(drop_out))
        print("Batch dim: " + str(batch_dim))
        print("Number of epochs: " + str(nn_epochs))
        print("\nLoss: " + loss + "\n")

        parallel_model.summary()

    return parallel_model


if __name__ == '__main__':
    print("This script contains the model")