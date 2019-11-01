import os
import numpy as np
from data_reader import data_reader

# --- reading in data ---
path_to_data = os.path.join('..', 'data', 'pa2_test_no_label.csv')
data = data_reader(path_to_data)


# --- reading in weights ---
path_to_weights = os.path.join('..', 'output', 'w_p1.csv')
weights_all = np.genfromtxt(path_to_weights, delimiter =',')

# --- testing model ---
iter_selection = 14								# iteration with highest accuracy
w = weights_all[iter_selection-1, :]			# selecting weights

y_hat = np.sign(w.dot(np.transpose(data)))								# testing model
file_out = os.path.join(os.getcwd(), '..', 'output', 'oplabel.csv') # path to output
np.savetxt(file_out, y_hat, fmt='%f', delimiter=',')				# writing output
