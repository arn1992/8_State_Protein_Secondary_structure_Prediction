
import numpy as np
from keras import optimizers, callbacks
from timeit import default_timer as timer
from data_2 import get_dataset, split_with_shuffle, get_data_labels, split_like_paper, get_cb513, get_casp10,get_casp11
import model
from time import time

do_log = True
stop_early = False
show_plots = True

dataset = get_dataset()

# D_train, D_test, D_val = split_like_paper(dataset)
#D_train, D_test, D_val = split_with_shuffle(dataset, 100)
D_train,  D_val = split_with_shuffle(dataset, 100)
X_train, Y_train = get_data_labels(D_train)
#X_test, Y_test = get_data_labels(D_test)
X_val, Y_val = get_data_labels(D_val)

net = model.CNN_model()

start_time = timer()

history = None

call_b = [model.checkpoint]

if do_log:
	# call_b.append(callbacks.TensorBoard(log_dir="../logs/Whole_CullPDB_Filtered/{}".format(time()), histogram_freq=0, write_graph=True))
    call_b.append(callbacks. TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=True))

if stop_early:
    call_b.append(model.early_stop)

history = net.fit(X_train, Y_train, epochs=model.nn_epochs, batch_size=model.batch_dim, shuffle=True,
                        validation_data=(X_val, Y_val), callbacks=call_b)

end_time = timer()
print("\n\nTime elapsed: " + "{0:.2f}".format((end_time - start_time)) + " s")



predictions1 = net.predict(X_train)
print("\n\nQ8 train accuracy: " + str(model.Q8_accuracy(Y_train, predictions1)) + "\n\n")
predictions2 = net.predict(X_val)
print("\n\nQ8 val accuracy: " + str(model.Q8_accuracy(Y_val, predictions2)) + "\n\n")
#predictions3 = net.predict(X_test)
#print("\n\nQ8 test accuracy: " + str(model.Q8_accuracy(Y_test, predictions3)) + "\n\n")

CB513_X, CB513_Y = get_cb513()

predictions = net.predict(CB513_X)

print("\n\nQ8 accuracy on CB513: " + str(model.Q8_accuracy(CB513_Y, predictions)) + "\n\n")

Casp10_X, Casp10_Y = get_casp10()

predictions8 = net.predict(Casp10_X)

print("\n\nQ8 accuracy on casp10: " + str(model.Q8_accuracy(Casp10_Y, predictions8)) + "\n\n")


Casp11_X, Casp11_Y = get_casp11()

predictions9 = net.predict(Casp11_X)

print("\n\nQ8 accuracy on casp11: " + str(model.Q8_accuracy(Casp11_Y, predictions9)) + "\n\n")


if show_plots:
    from plot import plot_history
    plot_history(history)

