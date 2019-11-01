import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from data_reader import data_reader




# --- reading in data ---
# -- training data
path_to_training = os.path.join('..', 'data', 'pa2_train.csv')
# data = data_reader(path_to_training)
data = np.genfromtxt(path_to_training, delimiter =',')
data[data[:,0] == 3, 0] = 1
data[data[:,0] == 5, 0] = -1
data = np.column_stack((data, np.ones(len(data))))

# -- validation data
path_to_validation = os.path.join('..', 'data', 'pa2_valid.csv')
# data_val = data_reader(path_to_validation)
data_val = np.genfromtxt(path_to_validation, delimiter =',')
data_val[data_val[:,0] == 3, 0] = 1
data_val[data_val[:,0] == 5, 0] = -1
data_val = np.column_stack((data_val, np.ones(len(data_val))))


iters = 15									# number of iterations
w = np.zeros(np.shape(data)[1] - 1)			# initializing weight vector
w_avg = np.zeros(np.shape(data)[1] - 1)		# initializing average weight vector

s = 1										# for running average calc.
acc_train = []								# preallocating list to save training accuracy
acc_val = []								# preallocating list to save valiodation accuracy
w_save = np.zeros((iters, len(w)))
for itr in range(iters):					# looping through iterations
	print('Iteration: {}' .format(itr))
	for i in range(len(data)):				# looping through data
		y_t = data[i,0]						# isolating y value for the data point
		x_t = data[i,1:]					# isolating x values for the data point
		if y_t*np.dot(w, x_t) <= 0:			# if there is an error:
			w += y_t*x_t					#	update the weight vector

		w_avg = (s*w_avg + w)/(s+1)			# keeping track of running weight 
		s += 1								# updating s for running average
	w_save[itr, :] = w_avg
	# --- testing results ---
	""" - compute w*x and take the sign of the resulting vector
		- the accuracy is then calculated by:
			- counting non-zero elements in the difference 
			  between the actual results and the predicted results.
			- the percentage is determined (e.g. number_of_incorrect/total)
			- the number of correct is then computed (e.g. 1-percent_incorrect)
	"""
	y_hat_training = np.sign(w_avg.dot(np.transpose(data[:,1:])))
	acc_train.append(1 - np.count_nonzero(data[:,0] - y_hat_training)/len(data))

	y_hat_training = np.sign(w_avg.dot(np.transpose(data_val[:,1:])))
	acc_val.append(1 - np.count_nonzero(data_val[:,0] - y_hat_training)/len(data_val))

# --- saving weights for each iteration to csv file ---
path_to_output = os.path.join(os.getcwd(), '..', 'output', 'w_p2.csv') 
np.savetxt(path_to_output, w_save, delimiter=',')


# --- plotting results --- 
iterations = np.linspace(1, iters, iters)

fig, ax = plt.subplots(1,1, figsize=(8, 6))
ax.plot(iterations, acc_train, color='k', ls='-', label = 'Training')
ax.plot(iterations, acc_val, color='k', ls='-.', label = 'Validation')

ax.legend()
ax.grid(which='minor', alpha=0.25, color = 'k', ls = ':')
ax.grid(which='major', alpha=0.40, color = 'k', ls = '--')
ax.set_xlabel('Iteration')
ax.set_ylabel('Accuracy')
plt.show()