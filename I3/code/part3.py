import os
import math
import numpy as np
import pandas as pd
import itertools
import HelperFunctions as HF


#adaboost model learner
#S: example features
#y: correct classification of example
#L: number of learners
def adaboost(data, features, y, L, depth):
    H = [] # weights for each feature, for each weak learner
    Alpha = [] # weight for each weak learner

    # N = len(data) # number of examples
    # print('# of examples: {0}'.format(N))

    # #initialize weights D, with uniform distribution initially
    # # D = [] #weights for each example, for each learner
    # # D.append(np.ones(N)/N)
    # data['D'] = np.ones(N)/N

    for l in range(L):
        # Dl = [] #empty D for D[l+1]
        h = learn(data, features, y, depth) #get model
        error, errlog = calcError(h,data,features,y) #find error of model
        alpha = .5*math.log((1 - error)/error) #calc weight
        #TODO: what happens if error = 0 or 1?, comments in slides say only if error<0.5

        #calculations for D[l+1]
        i = 0 #example index
        for index,ex in data.iterrows():
        # for err in errlog: 
            # Dex = 0
            if (errlog[i]): #guessed wrong
                # Dex = D[l][i]*math.exp(alpha)
                data['D'][index] = ex['D']*math.exp(alpha)
            else: #guessed right
                # Dex = D[l][i]*math.exp(-alpha)
                data['D'][index] = ex['D']*math.exp(-alpha)

            # Dl.append(Dex) #save D[l+1][ex]
            i += 1 #increment for next loop

        #normalize D
        print('Sum of D before: {0}'.format(data.sum(axis=0)['D']))
        Norm = data.sum(axis=0)['D'] #TODO: make sure this is how you normalize
        data['D'] = data['D']/Norm
        print('Sum of D after: {0}'.format(data.sum(axis=0)['D']))
        # D.append(Dl) #save D[l+1]
        H.append(h)  #save model for learner
        Alpha.append(alpha) #save vote weight for learner

    return H, Alpha


#weak learner (DT with depth and weighted examples)
def learn(data,features,y,depth):
    h = []

    N = len(data) # number of examples
    print('# of examples: {0}'.format(N))

    #initialize weights D, with uniform distribution initially
    # D = [] #weights for each example, for each learner
    # D.append(np.ones(N)/N)
    data['D'] = np.ones(N)/N

    c0 = len(data.loc[data[y] == -1])
    c1 = len(data.loc[data[y] == 1])
    sumD1 = sum(data.loc[data[y] == 1]['D']) #0
    sumD0 = sum(data.loc[data[y] == -1]['D']) # 0
    # for i in range(len(data)):
    #     if data[i][y] == 1:
    #         sumD1 += data[i]['D']
    #     else:
    #         sumD0 += data[i]['D']
    

    p1 = (sumD1/(sumD1+sumD0))				# probabilty that y=1
    p0 = (sumD0/(sumD1+sumD0))				# probabilty that y=0

    tree = {}
    tree['root'] = {}
    tree['root']['data'] = data
    tree['root']['f=0'] = c0
    tree['root']['f=1'] = c1
    tree['root']['prob'] = 1
    tree['root']['U'] = 1 - p1**2 - p0**2				#note: want to confirm this.
    tree['root']['continue'] = True
    tree['root']['split_on'] = None
    tree['root']['p1'] = p1
    tree['root']['p0'] = p0


    for i in range(depth):
        temp = list(itertools.product([1,0], repeat=i+1))

        if len(temp[0]) == 1:
            temp = [str(i[0]) for i in temp]
        else:
            temp = ['-'.join(map(str, i)) for i in temp]
        for ii in temp:
            tree[ii] = {}
            tree[ii]['data'] = None
            tree[ii]['f=0'] = None
            tree[ii]['f=1'] = None
            tree[ii]['prob'] = None
            tree[ii]['U'] = None
            tree[ii]['continue'] = None
            tree[ii]['split_on'] = None
            tree[ii]['p1'] = None
            tree[ii]['p0'] = None

    for i in range(1,depth+1):
        if (len(features)) > 0:
            tree, features = HF.find_best_feature(tree, layer=i, features=features, y=y, mode='adaboost')

    tree = {i:j for i,j in tree.items() if j != {}}	# removing empty keys


    return h


#calculate the weighted error, returns a log of the results too
#for the log, 0 if guess is correct, 1 if wrong
def calcError(h,data,features,y):
    errlog = []
    error = 0

    # i = 0 #example index
    for index,ex in data.iterrows():
        # print('\n\n')
        # print(ex)
        if (useModel(h,ex,features) == ex[y]):
            errlog.append(0) 
        else:
            errlog.append(1)
            error += ex['D']
        # i += 1
    print('Total number of errors: {0}'.format(error))
    return error, errlog


#using a single learned model of adaboost
#returns 1 if poisonous
def useModel(h,ex,features):
    isPoisonous = -1

    return isPoisonous


#using the adaboost model
#returns 1 if poisonous
def toEatOrNotToEat(learners,weight,features):
    isPoisonous = -1

    return isPoisonous



###### RUN HERE ############################################################
path_to_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
# data = HF.datareader(path_to_data)
# S = data.iloc[:, :-1] # features for each example
# y = data.iloc[:,-1] # mushroom classes

data = HF.datareader(path_to_data)
features= list(data.columns[:-1])
y = data.columns[-1]
print(data['cap-shape_c'][1])
# print(features)
# print(y)

H_all = [] #all the models
Alpha_all = [] #all the vote weights

# Part C
# Depth of 1
learners = [1,2,5,10,15] #number of learners to try
for L in learners:
    # H, Alpha = adaboost(S,y,L,1) #run adaboost
    H, Alpha = adaboost(data,features,y,L,1) #run adaboost
    #save results
    H_all.append(H)
    Alpha_all.append(Alpha)

# Part E
# learners = [1,2,5,10,15] #number of learners to try
# for L in learners:
#     H, Alpha = adaboost(S,y,L) #run adaboost
#     #save results
#     H_all.append(H)
#     Alpha_all.append(Alpha)
