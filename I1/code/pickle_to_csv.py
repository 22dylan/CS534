import os
import numpy as np
import pickle

from data_reader import data_reader
from data_reader import results_to_csv


def pickle_to_csv(results_name):
    pickle_name = "{0}.pickle".format(results_name)

    path_to_pickle = os.path.join(os.getcwd(), '..', 'output', pickle_name)				# path to output
    data = pickle.load(open(path_to_pickle, "rb" ))

    results_to_csv(results_name, data)


pickle_to_csv("results_p1")