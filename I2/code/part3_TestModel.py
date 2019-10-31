import os
import numpy as np
from data_reader import data_reader
from part3_TrainModel import gramMatrix

def countErrs(alpha, y):
    err = []
    for run in range(len(alpha)):
        err.append(np.dot(alpha.T,y))

    return err

#get alphas
path_to_alpha = os.path.join('..', 'output', 'alpha3_p3.csv')
alpha_all = np.genfromtxt(path_to_alpha, delimiter =',')
alpha = alpha_all[14]

#get test data
path_to_training = os.path.join('..', 'data', 'pa2_test.csv')
data_test = np.genfromtxt(path_to_training, delimiter =',')
x_test = np.column_stack((data_test, np.ones(len(data_test))))

K,N = gramMatrix(x,p) 





