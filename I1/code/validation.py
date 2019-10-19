import os
import pandas as pd
import pickle
import numpy as np

from data_reader import data_reader
from data_reader import save_results

#function to calculate sum squared error of a given
#weight vector (w) and data matrix where the last column
#is the price that w is estimating 
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

def validate_W(part_num):
    #prevent numpy arrays from wrapping to new line
    np.set_printoptions(linewidth=np.inf) 

    path_to_data = os.path.join(os.getcwd(), '..', 'data')	# path to data
    path_to_output = os.path.join(os.getcwd(), '..', 'output')	# path to output

    #filenames
    train_results_pickle_nm = 'results_{0}.pickle'.format(part_num)
    validation_results_pickle_nm = 'results_validation_{0}.pickle'.format(part_num)
    path_to_pickle = os.path.join(path_to_output, validation_results_pickle_nm)

    # reading in data
    normval = os.path.join(path_to_output, 'normalizing_values.csv')
    normval = pd.read_csv(normval).set_index('Unnamed: 0').transpose()

    validation_data = os.path.join(path_to_data, 'PA1_dev.csv')
    validation_data, normval = data_reader(validation_data,norm=True, normval=normval)		# reading in normalized data

    part_pickle = os.path.join(path_to_output, train_results_pickle_nm) #get file name of training results

    results = pickle.load(open(part_pickle, "rb" )) #load training results

    validation_SSE = {}
    for key, trial in results.items():
        w = trial['w_values']
        validation_SSE[key] = calcSSE(w, validation_data) 

    # print(validation_SSE)

    path_to_pickle = os.path.join(path_to_output, validation_results_pickle_nm)

    # saving results to a pickle
    with open(path_to_pickle, 'wb') as f:
        pickle.dump(validation_SSE, f, pickle.HIGHEST_PROTOCOL)

    # print(validation_SSE)

validate_W("p1")