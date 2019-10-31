import os
import numpy as np
import math
# import matplotlib.pyplot as plt
import pandas as pd
from data_reader import data_reader

def gramMatrix(x,p):
    N = len(x)
    K = []
    for j in range(N):
        row = []
        for i in range(N):
            xtx = np.dot(x[i].T,x[j])
            kern = math.pow( (1 + xtx) ,p)
            row.append(kern)
        K.append(row)
    
    return K,N

#part 3 perceptron algorithm with kernel fn
def kernelPerceptron(data,data_val, p):
    iterations = 15
    x = data[:,1:] #isolate x
    x_val = data_val[:,1:]

    y = data[:,0] #isolate y
    y_val = data_val[:,0]

    #compute gram matrix and NxN dimension
    #K[j][i] where j is row, i is column
    K,N = gramMatrix(x,p) 
    K_val, N_val = gramMatrix(x_val,p)

    alpha = np.zeros((iterations,N))
    err_train = []
    err_val = []

    for run in range(iterations):
        print('Starting run {0}'.format(run+1))

        #update alpha from previous iteration
        if run>0:
            alpha[run] = alpha[run-1]

        err = 0
        for j in range(N):
            
            u = 0
            #calculate u from summing alpha*K*y
            for i in range(N):
                u = u + (alpha[run][i] * K[j][i] * y[i])
            
            #if guess is wrong, update alpha
            if y[j]*u <= 0:
                alpha[run][j] = alpha[run][j] + 1
                err = err + 1
        err_train.append(err) #save number of errors of 

        #test against validation data
        err2 = 0
        for j in range(N_val):
            u_val = 0
            for i in range(N_val):
                u_val = u_val + (alpha[run][i] * K_val[j][i] * y_val[i])
            
            if y_val[j]*u_val <= 0:
                alpha[run][j] = alpha[run][j] + 1
                err2 = err2 + 1
        err_val.append(err2)

    return alpha, err_train, err_val


def runPerceptrons():
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

    err_train_all = []
    err_val_all = []
    #run perceptron
    for p in [1,2,3,4,5]:
        print("P: {0}".format(p))
        alpha, err_train, err_val = kernelPerceptron(data, data_val, p)
        print("err_train")
        print(err_train)
        print("err_val")
        print(err_val)

        err_train_all.append(err_train)
        err_val_all.append(err_val)

        #save results
        path_to_output = os.path.join(os.getcwd(), '..', 'output', 'alpha{0}_p3.csv'.format(p)) 
        np.savetxt(path_to_output, alpha, delimiter=',')

    #save results
    path_to_output = os.path.join(os.getcwd(), '..', 'output', 'err_train_p3.csv') 
    np.savetxt(path_to_output, err_train_all, delimiter=',')

    #save results
    path_to_output = os.path.join(os.getcwd(), '..', 'output', 'err_val_p3.csv') 
    np.savetxt(path_to_output, err_val_all, delimiter=',')


runPerceptrons()

