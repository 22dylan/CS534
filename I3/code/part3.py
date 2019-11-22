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

    N = len(S) # number of examples

    #initialize weights D, with uniform distribution initially
    D = [] #weights for each example, for each learner
    D.append(np.ones(N)/N)

    for l in range(L):
        Dl = [] #empty D for D[l+1]
        h = learn(S,y,D[l]) #get model
        error, errlog = calcError(h,S,y,D[l]) #find error of model
        alpha = .5*math.log((1 - error)/error) #calc weight

        #calculations for D[l+1]
        i = 0 #example index
        for err in errlog: 
            Dex = 0
            if (err): #guessed wrong
                Dex = D[l][i]*math.exp(alpha)
            else: #guessed right
                Dex = D[l][i]*math.exp(-alpha)

            Dl.append(Dex) #save D[l+1][ex]
            i += 1 #increment for next loop

        #normalize D
        print('Sum of D before: {0}'.format(sum(Dl)))
        Norm = sum(Dl) #TODO: make sure this is how you normalize
        Dl = Dl/Norm
        print('Sum of D after: {0}'.format(sum(Dl)))
        D.append(Dl) #save D[l+1]
        H.append(h)  #save model for learner
        Alpha.append(alpha) #save vote weight for learner
    return H, Alpha


#weak learner (DT with depth 1 and weighted examples)
def learn(S,y,D):
    h = []

    return h


#calculate the weighted error, returns a log of the results too
#for the log, 0 if guess is correct, 1 if wrong
def calcError(h,S,y,D):
    errlog = []
    error = 0

    i = 0 #example index
    for ex in S:
        if (useModel(h,ex) == y[i]):
            errlog.append(0) 
        else:
            errlog.append(1)
            error += D[i]
        i += 1

    return error, errlog


#using a single learned model of adaboost
#returns 1 if poisonous
def useModel(h,features):
    isPoisonous = 0

    return isPoisonous


#using the adaboost model
#returns 1 if poisonous
def toEatOrNotToEat(h,weight,features):
    isPoisonous = 0

    return isPoisonous



###### RUN HERE ############################################################
path_to_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
data = HF.datareader(path_to_data)
S = data.iloc[:, :-1] # features for each example
y = data.iloc[:,-1] # mushroom classes

H_all = [] #all the models
Alpha_all = [] #all the vote weights
learners = [1,2,5,10,15] #number of learners to try
for L in learners:
    H, Alpha = adaboost(S,y,L) #run adaboost
    #save results
    H_all.append(H)
    Alpha_all.append(Alpha)