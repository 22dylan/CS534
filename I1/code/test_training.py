import os
import pandas as pd
import pickle

from data_reader import data_reader


path_to_data = os.path.join(os.getcwd(), '..', 'data')	# path to data
path_to_output = os.path.join(os.getcwd(), '..', 'output')				# path to output

normval = os.path.join(path_to_output, 'normalizing_values.csv')
normval = pd.read_csv(normval).set_index('Unnamed: 0').transpose()

test_data = os.path.join(path_to_data, 'PA1_test.csv')
test_data, normval = data_reader(test_data,norm=True, normval=normval)		# reading in normalized data


part_1_pickle = os.path.join(path_to_output, 'results_p1.pickle')
part_2_pickle = os.path.join(path_to_output, 'results_p2.pickle')
part_3_pickle = os.path.join(path_to_output, 'results_p3.pickle')

results_p1 = pickle.load(open(part_1_pickle, "rb" ))
results_p2 = pickle.load(open(part_2_pickle, "rb" ))
# results_p3 = pickle.load(open(part_3_pickle, "rb" ))

w_p1 = results_p1['trial_0']['w_values']


