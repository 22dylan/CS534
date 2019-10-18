import os
import numpy as np
import csv

# test = np.random.rand(5,5)
# test = np.full((50,4),.5)

def bgd(lambda_reg, stepSize, data):
    
    iterations = 0
    SSE_results = []

    #initialize w to number of parameters, ignoring cost column
    params_len = data[0].size - 1
    w = np.zeros(params_len) #initial guess 

    #start with gradient of loss above .5
    loss_gradient = np.full(params_len,100) 

    #repeat until convergence
    while (np.linalg.norm(loss_gradient) >= .5):
        
        sumErr = np.zeros(params_len) #init to zero
        for row in data:
            # print(row)
            y_i = row[params_len]
            x_i = row[0:params_len]
            # print(y_i)
            # print(x_i)
            sumErr = sumErr + (y_i - np.dot(w.T,x_i))*x_i

        loss_gradient = (-2)*sumErr + 2*lambda_reg*w
        w = w - stepSize*loss_gradient

        iterations = iterations + 1

        #save sum square error every 100 iterations
        if (iterations%100 == 0):
            SSE_results.append([iterations, np.linalg.norm(sumErr)])

        #print info to know its working
        if (iterations%1000 == 0):
            print("Done with iteration {0}".format(iterations))
            print("|loss_gradient|: {0}".format(np.linalg.norm(loss_gradient)))
            print("y_i: {0}".format(y_i))
            print("w.T*x_i: {0}".format(np.dot(w.T,x_i)))
            # print("w: {0}".format(w))
            # print("x_i: {0}".format(x_i))
            # print("sumErr: {0}".format(sumErr))
            print("")
            # input()
        
        #stop if diverging
        if (np.linalg.norm(loss_gradient) > 10**30):
            break

    print("Regression complete with iteration: {0}".format(iterations))
    print("|loss_gradient|: {0}".format(np.linalg.norm(loss_gradient)))
    print("SSE_results: {0}".format(SSE_results))
    print("w: {0}".format(w))
    # print("x_i: {0}".format(x_i))
    print("y_i: {0}".format(y_i))
    print("w.T*x_i: {0}".format(np.dot(w.T,x_i)))
    # print("sumErr: {0}".format(sumErr))
    print("")
    # input()
    return w, iterations, SSE_results

# bgd(0, .01, test)

