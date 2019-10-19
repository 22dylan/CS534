import os
import pandas as pd
import pickle
import numpy as np

from data_reader import data_reader
from data_reader import save_results


def calcSSE(w, data):
    #initialize w to number of parameters, ignoring cost column
    params_len = data[0].size - 1

    sumErr = np.zeros(params_len) #init to zero
    for row in data:
        # print(row)
        y_i = row[params_len]
        x_i = row[0:params_len]
        # print(y_i)
        # print(x_i)
        sumErr = sumErr + (y_i - np.dot(w.T,x_i))*x_i

    return sumErr

#prevent numpy arrays from wrapping to new line
np.set_printoptions(linewidth=np.inf) 

path_to_data = os.path.join(os.getcwd(), '..', 'data')	# path to data
path_to_output = os.path.join(os.getcwd(), '..', 'output')				# path to output

# reading in data
normval = os.path.join(path_to_output, 'normalizing_values.csv')
normval = pd.read_csv(normval).set_index('Unnamed: 0').transpose()

validation_data = os.path.join(path_to_data, 'PA1_dev.csv')
validation_data, normval = data_reader(validation_data,norm=True, normval=normval)		# reading in normalized data

part_1_pickle = os.path.join(path_to_output, 'results_p1.pickle')
# part_2_pickle = os.path.join(path_to_output, 'results_p2.pickle')
# part_3_pickle = os.path.join(path_to_output, 'results_p3.pickle')

results_p1 = pickle.load(open(part_1_pickle, "rb" ))
# results_p2 = pickle.load(open(part_2_pickle, "rb" ))
# results_p3 = pickle.load(open(part_3_pickle, "rb" ))

validation_SSE = {}
for key, trial in results_p1.items():
    w = trial['w_values']
    validation_SSE[key] = calcSSE(w, validation_data) 


path_to_pickle_p1 = os.path.join(path_to_output, 'results_validation_p1.pickle')

# saving results to a pickle
with open(path_to_pickle_p1, 'wb') as f:
    pickle.dump(validation_SSE, f, pickle.HIGHEST_PROTOCOL)

# print(validation_SSE)
