import os
import numpy as np
import csv

# test = np.random.rand(5,5)
# test = np.full((50,4),.5)

def bgd(lambda_reg, stepSize, data):

    #initialize w to number of parameters, ignoring cost column
    params_len = data[0].size - 1
    w = np.zeros(params_len) #initial guess 

    #start with gradient of loss above .5
    loss_gradient = np.full(params_len,1) 

    #repeat until convergence
    while (np.linalg.norm(loss_gradient) >= 0.5):
        sumErr = np.zeros(params_len) #init to zero
        for row in data:
            # print(row)
            y_i = row[params_len]
            x_i = row[0:params_len]
            # print(y_i)
            # print(x_i)
            sumErr = sumErr + (y_i - np.dot(w.T,x_i))*x_i

            #debug info
            # print("w: {0}".format(w))
            # print("x_i: {0}".format(x_i))
            # print("y_i: {0}".format(y_i))
            # print("w.T*x_i: {0}".format(np.dot(w.T,x_i)))
            # print("sumErr: {0}".format(sumErr))
            # print("")

        loss_gradient = (-2)*sumErr + 2*lambda_reg*w
        w = w - stepSize*loss_gradient
        
        # print("w: {0}".format(w))
        # print("")
        # input()


    return w

# bgd(0, .01, test)

