import numpy as np
from time import time
from keras import optimizers, callbacks
from timeit import default_timer as timer
from data_1 import get_dataset_reshaped, split_dataset, get_resphaped_dataset_paper, get_cb513, is_filtered,get_casp10,get_casp11
import model_1

import pickle

filtered = is_filtered()

do_log = True
stop_early = False
show_plots = True

start_time = timer()

print("Collecting Dataset...")

if filtered:
    # Split the dataset in 0.8 train, 0.1 test, 0.1 validation with shuffle (optionally seed)
    #X_train, X_val, X_test, Y_train, Y_val, Y_test = get_dataset_reshaped(seed=100)
    X_train, X_val,  Y_train, Y_val = get_dataset_reshaped(seed=100)

else:
    # Slit the dataset with the same indexes used in the paper (Only for CullPDB6133 not filtered)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = get_resphaped_dataset_paper()

end_time = timer()
print("\n\nTime elapsed getting Dataset: " + "{0:.2f}".format((end_time - start_time)) + " s")

if filtered:
    print("Using CullPDB Filtered dataset")

net = model_1.CNN_model()

start_time = timer()

history = None

call_b = [model_1.checkpoint]

if filtered:
    logDir = "logs/CullPDB_Filtered/{}".format(time())
else:
    logDir = "logs/CullPDB/{}".format(time())

if do_log:
    call_b.append(callbacks.TensorBoard(log_dir=logDir, histogram_freq=0, write_graph=True))

if stop_early:
    call_b.append(model_1.early_stop)

history = net.fit(X_train, Y_train, epochs=model_1.nn_epochs, batch_size=model_1.batch_dim, shuffle=True,
                        validation_data=(X_val, Y_val), callbacks=call_b)

end_time = timer()
print("\n\nTime elapsed: " + "{0:.2f}".format((end_time - start_time)) + " s")

#scores = net.evaluate(X_test, Y_test)
#print("Loss: " + str(scores[0]) + ", Accuracy: " + str(scores[1]) + ", MAE: " + str(scores[2]))
#print(scores)

CB_x, CB_y = get_cb513()

cb_scores = net.evaluate(CB_x, CB_y)
print("CB513 -- Loss: " + str(cb_scores[0]) + ", Accuracy: " + str(cb_scores[1]) + ", MAE: " + str(cb_scores[2]))


CB_x1, CB_y1 = get_casp10()

cb_scores1 = net.evaluate(CB_x1, CB_y1)
print("Casp10 -- Loss: " + str(cb_scores1[0]) + ", Accuracy: " + str(cb_scores1[1]) + ", MAE: " + str(cb_scores1[2]))


CB_x2, CB_y2 = get_casp11()

cb_scores2 = net.evaluate(CB_x2, CB_y2)
print("Casp11 -- Loss: " + str(cb_scores2[0]) + ", Accuracy: " + str(cb_scores2[1]) + ", MAE: " + str(cb_scores2[2]))


pickle_out = open("lasthistory.pickle","wb")
pickle.dump(history, pickle_out)
pickle_out.close()

if show_plots:
    from plot_history import plot_history
    plot_history(history)