

import numpy as np
from keras import optimizers, callbacks
from timeit import default_timer as timer
from data_2 import get_dataset, split_with_shuffle, get_data_labels, split_like_paper, get_cb513
import model

dataset = get_dataset()
#dataset_1=get_dataset_1()

D_train, D_val = split_with_shuffle(dataset, 100)

X_train, Y_train = get_data_labels(D_train)
X_test, Y_test = get_data_labels(D_test)
X_val, Y_val = get_data_labels(D_val)

#D_train_1 = split_with_shuffle_1(dataset_1, 100)

#X_train_1, Y_train_1 = get_data_labels(D_train_1)
#X_test_1, Y_test_1 = get_data_labels(D_test_1)
#X_val_1, Y_val_1 = get_data_labels(D_val_1)

net = model.CNN_model()

#load Weights
net.load_weights("Whole_CullPDB-best.hdf5")
'''
predictions1 = net.predict(X_train)
print("\n\nQ8 train accuracy: " + str(model.Q8_accuracy(Y_train, predictions1)) + "\n\n")
predictions2 = net.predict(X_val)
print("\n\nQ8 val accuracy: " + str(model.Q8_accuracy(Y_val, predictions2)) + "\n\n")
#predictions3 = net.predict(X_test)
#print("\n\nQ8 test accuracy: " + str(model.Q8_accuracy(Y_test, predictions3)) + "\n\n")

predictions4 = net.predict(X_train_1)
print("\n\nQ8 test accuracy for filtered: " + str(model.Q8_accuracy(Y_train_1, predictions4)) + "\n\n")


#predictions5 = net.predict(X_val_1)
#print("\n\nQ8 val accuracy for filtered: " + str(model.Q8_accuracy(Y_test, predictions5)) + "\n\n")
'''

CB513_X, CB513_Y = get_cb513()

predictions = net.predict(CB513_X)

print("\n\nQ8 accuracy on CB513: " + str(model.Q8_accuracy(CB513_Y, predictions)) + "\n\n")

Casp10_X, Casp10_Y = get_cb513()

predictions8 = net.predict(Casp10_X)

print("\n\nQ8 accuracy on casp10: " + str(model.Q8_accuracy(Casp10_Y, predictions8)) + "\n\n")

