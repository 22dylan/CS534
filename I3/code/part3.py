import os
import math
import numpy as np
import pandas as pd
import HelperFunctions as HF


#adaboost model learner
#S: example features
#y: correct classification of example
#L: number of learners
def adaboost(S, y, L):
    H = [] # weights for each feature, for each weak learner
    Alpha = [] # weight for each weak learner

    
    N = len(S[0]) # number of features for each example

    #initialize weights D, with uniform distribution initially
    D = [] #weights for each example, for each learner
    D.append(np.ones((1,N))/N)

    for l in range(L):
        Dl = [] #empty D for D[l+1]
        h = learn(S,y,D[l]) #get model
        error = error(S,y,D[l]) #find error of model
        alpha = .5*math.log((1 - error)/error) #calc weight

        #calculations for D[l+1]
        i = 0 #example index
        for ex in S: 
            Dex = 0
            if (h*ex == y[i]):
                Dex = D[l][i]*math.exp(alpha)
            else:
                Dex = D[l][i]*math.exp(-alpha)
            #normalize D
            Norm = sum(Dex) #TODO: make sure this is how you normalize
            Dex = Dex/Norm
            Dl.append(Dex) #save D[l+1][ex]
            i += 1 #increment for next loop

        D.append(Dl) #save D[l+1]
        H.append(h)  #save model for learner
        Alpha.append(alpha) #save vote weight for learner
    return H, Alpha


#weak learner (DT with depth 1 and weighted examples)
def learn(S,y,D):
    h = []


    return h


#calculate the weighted error
def error(S,y,D):
    error = 0


    return error


#using the adaboost model
def toEatOrNotToEat(S):
    pass



###### RUN HERE ############################################################
path_to_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
data = HF.datareader(path_to_data)
S = list(data.columns[:-1]) # features for each example
y = data.columns[-1] # mushroom classes

H_all = [] #all the models
Alpha_all = [] #all the vote weights
learners = [1,2,5,10,15] #number of learners to try
for L in learners:
    H, Alpha = adaboost(S,y,L) #run adaboost
    #save results
    H_all.append(H)
    Alpha_all.append(Alpha)