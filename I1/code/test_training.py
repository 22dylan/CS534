import os
import pandas as pd
import pickle
import numpy as np

from data_reader import data_reader
from data_reader import price_pickle_to_csv

def test_training(part_num, trial_num):
    path_to_data = os.path.join(os.getcwd(), '..', 'data')	# path to data
    path_to_output = os.path.join(os.getcwd(), '..', 'output')				# path to output

    #filenames
    train_results_pickle_nm = 'results_{0}.pickle'.format(part_num)
    test_results_pickle_nm = 'test_{0}.pickle'.format(part_num)
    path_to_pickle = os.path.join(path_to_output, test_results_pickle_nm)

    # reading in data
    normval = os.path.join(path_to_output, 'normalizing_values.csv')
    normval = pd.read_csv(normval).set_index('Unnamed: 0').transpose()

    test_data = os.path.join(path_to_data, 'PA1_test.csv')
    test_data, normval = data_reader(test_data,norm=True, normval=normval)		

    part_pickle = os.path.join(path_to_output, train_results_pickle_nm)
    results = pickle.load(open(part_pickle, "rb" ))

    # predicted_price = {}
    w = results[trial_num]['w_values']
    predicted_price = test_data.dot(w) 

    print(normval['price'][0])
    #un-normalize prices
    predicted_price = predicted_price*normval['price'][0]

    print(predicted_price)
    # saving results to a pickle
    with open(path_to_pickle, 'wb') as f:
        pickle.dump(predicted_price, f, pickle.HIGHEST_PROTOCOL)

    price_pickle_to_csv(part_num,predicted_price)


test_training("p1","trial_6")