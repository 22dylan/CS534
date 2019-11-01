import os
import numpy as np
from data_reader import data_reader
from part3_TrainModel import gramMatrix


#get alphas
path_to_alpha = os.path.join('..', 'output', 'alpha3_p3.csv')
alpha_all = np.genfromtxt(path_to_alpha, delimiter =',')
alpha = alpha_all[3] #p=3, iter=4


# get training data
path_to_training = os.path.join('..', 'data', 'pa2_train.csv')
data = np.genfromtxt(path_to_training, delimiter =',')
data[data[:,0] == 3, 0] = 1
data[data[:,0] == 5, 0] = -1
data = np.column_stack((data, np.ones(len(data))))
x = data[:,1:] #isolate x
y = data[:,0] #isolate y

#get test data
path_to_training = os.path.join('..', 'data', 'pa2_test_no_label.csv')
data_test = np.genfromtxt(path_to_training, delimiter =',')
x_test = np.column_stack((data_test, np.ones(len(data_test))))

K, N, M = gramMatrix(x_test,x,3)
y_guess = []

for j in range(M):   
    u = 0
    #calculate u from summing alpha*K*y
    for i in range(N):
        u = u + (alpha[i] * K[j][i] * y[i])
    y_guess.append((np.sign(u)))

# print(y_guess)
#save results
path_to_output = os.path.join(os.getcwd(), '..', 'output', 'kplabel.csv') 
np.savetxt(path_to_output, y_guess, delimiter=',', fmt='%f')

            







