import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from data_reader import data_reader

def gramMatrix(xdata,xt,p):
    N = len(xt)
    M = len(xdata)
    K = []
    for j in range(M):
        row = []
        for i in range(N):
            xtx = np.dot(xdata[j].T,xt[i])
            kern = math.pow( (1 + xtx) ,p)
            row.append(kern)
        K.append(row)
    
    return K,N,M

#part 3 perceptron algorithm with kernel fn
def kernelPerceptron(data,data_val, p):
    iterations = 15
    x = data[:,1:] #isolate x
    x_val = data_val[:,1:]

    y = data[:,0] #isolate y
    y_val = data_val[:,0]

    #compute gram matrix and NxN dimension
    #K[j][i] where j is row, i is column
    K,N,M = gramMatrix(x,x,p) 
    K_val, N_val, M_val = gramMatrix(x_val,x,p)

    alpha = np.zeros((iterations,N))
    acc_train = []
    acc_val = []

    for run in range(iterations):
        print('Starting run {0}'.format(run+1))

        #update alpha from previous iteration
        if run>0:
            alpha[run] = alpha[run-1]

        err = 0
        for j in range(M):
            
            u = 0
            #calculate u from summing alpha*K*y
            for i in range(N):
                u = u + (alpha[run][i] * K[j][i] * y[i])
            
            #if guess is wrong, update alpha
            if y[j]*u <= 0:
                alpha[run][j] = alpha[run][j] + 1
                err = err + 1
        acc_train.append(round(1- (err/M),5)) #save number of errors of 

        #test against validation data
        err2 = 0
        for j in range(M_val):
            u_val = 0
            for i in range(N_val):
                u_val = u_val + (alpha[run][i] * K_val[j][i] * y[i])
            
            if y_val[j]*u_val <= 0:
                err2 = err2 + 1
        acc_val.append(round(1- (err2/M_val),5))

    return alpha, acc_train, acc_val


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

    acc_train_all = []
    acc_val_all = []
    #run perceptron
    for p in [1,2,3,4,5]:
        print("P: {0}".format(p))
        alpha, acc_train, acc_val = kernelPerceptron(data, data_val, p)
        print("acc_train")
        print(acc_train)
        print("acc_val")
        print(acc_val)

        acc_train_all.append(acc_train)
        acc_val_all.append(acc_val)

        #save results
        path_to_output = os.path.join(os.getcwd(), '..', 'output', 'alpha{0}_p3.csv'.format(p)) 
        np.savetxt(path_to_output, alpha, delimiter=',')

    #save results
    path_to_output = os.path.join(os.getcwd(), '..', 'output', 'acc_train_p3.csv') 
    np.savetxt(path_to_output, acc_train_all, delimiter=',')

    #save results
    path_to_output = os.path.join(os.getcwd(), '..', 'output', 'acc_val_p3.csv') 
    np.savetxt(path_to_output, acc_val_all, delimiter=',')

def plotData():
    #get accuracy
    path_to_acc_train = os.path.join('..', 'data', 'acc_train_p3.csv')
    acc_train_all = np.genfromtxt(path_to_acc_train, delimiter =',')

    path_to_acc_val = os.path.join('..', 'data', 'acc_val_p3.csv')
    acc_val_all = np.genfromtxt(path_to_acc_val, delimiter =',')

    iterations = np.linspace(1, 15, 15)

    fig, ax = plt.subplots(1,1, figsize=(8, 6))
    p = 0
    for acc_train in acc_train_all:
        ax.plot(iterations, acc_train, color='k', ls='-', label = 'Training, p={0}'.format(p+1))
        p += 1
    p = 0
    for acc_val in acc_val_all:
        ax.plot(iterations, acc_val, color='k', ls='-.', label = 'Validation, p={0}'.format(p+1))
        p += 1

    ax.legend()
    ax.grid(which='minor', alpha=0.25, color = 'k', ls = ':')
    ax.grid(which='major', alpha=0.40, color = 'k', ls = '--')
    plt.show()


runPerceptrons()
# plotData()
