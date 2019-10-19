import os
import numpy as np
import pickle

from data_reader import data_reader
from data_reader import results_to_csv
from data_reader import validation_to_csv


def pickle_to_csv(results_name):
    pickle_name = "{0}.pickle".format(results_name)

    path_to_pickle = os.path.join(os.getcwd(), '..', 'output', pickle_name)				# path to output
    data = pickle.load(open(path_to_pickle, "rb" ))

    results_to_csv(results_name, data)

def val_pickle_to_csv(validation_name):
    pickle_name = "{0}.pickle".format(validation_name)

    path_to_pickle = os.path.join(os.getcwd(), '..', 'output', pickle_name)				# path to output
    data = pickle.load(open(path_to_pickle, "rb" ))

    validation_to_csv(validation_name, data)


# pickle_to_csv("results_p1")
# pickle_to_csv("results_p2_drs")
# pickle_to_csv("results_p2_0to4")
# pickle_to_csv("results_p2_2ndRun")
# pickle_to_csv("results_p3")

val_pickle_to_csv("results_validation_p1")
# val_pickle_to_csv("results_validation_p2_drs")
# val_pickle_to_csv("results_validation_p2_0to4")
# val_pickle_to_csv("results_validation_p2_2ndRun")
# val_pickle_to_csv("results_validation_p3")