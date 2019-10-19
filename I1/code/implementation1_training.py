import os
import numpy as np
import pandas as pd
import pickle

from run_bgd import run_bgd
from data_reader import data_reader


# setting up paths
path_to_data = os.path.join(os.getcwd(), '..', 'data', 'PA1_train.csv')	# path to data
path_to_output = os.path.join(os.getcwd(), '..', 'output')				# path to output

# reading in data and saving normalizing values
print('Reading Data')

normed_data, normval = data_reader(path_to_data,norm=True)		# reading in normalized data
raw_data, _ = data_reader(path_to_data,norm=False)				# reading in raw data

normalizing_values_out = os.path.join(path_to_output, 'normalizing_values.csv')	# path to save normalized values
pd.DataFrame(normval).to_csv(normalizing_values_out)			# saving normalizing values

print('---------------')

# # --- training part 1 --- 
# print('Training with different learning rates')

# step_size = [10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7]	# learning rates or step sizes
# lambda_vals = [0]				# regularization parameter set to 0
# path_to_pickle = os.path.join(path_to_output, 'results_p1.pickle')		# output pickle

# results_p1 = run_bgd(normed_data, step_size, lambda_vals, path_to_pickle)	# running batch gradient descent with above values
# print('---------------')


# --- training part 2 ---
""" description here """
print('Training with different learning rates and regularization paramters')
step_size = [10**-5, 10**-6, 10**-7]
lambda_vals = [0, 10**-3, 10**-2, 10**-1, 1, 10, 100]
path_to_pickle = os.path.join(path_to_output, 'results_p2_drs.pickle')

results_p2 = run_bgd(normed_data, step_size, lambda_vals, path_to_pickle)
print('---------------')

# # --- training part 3 ---
# print('Training on non-normalized data')
# step_size = [10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7]
# lambda_vals = [0, 10**-3, 10**-2, 10**-1, 1, 10, 100]
# path_to_pickle = os.path.join(path_to_output, 'results_p3.pickle')

# results_p3 = run_bgd(raw_data, step_size, lambda_vals, path_to_pickle)


# print('---------------')
