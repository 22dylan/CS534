import os
import math
import copy
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
    # data = data1
    N = len(data) # number of examples
    # print('# of examples: {0}'.format(N))

    # #initialize weights D, with uniform distribution initially
    # D = [] #weights for each example, for each learner
    # D.append(np.ones(N)/N)
    data['D'] = np.ones(N)/N

    for l in range(L):
        # Dl = [] #empty D for D[l+1]
        h = HF.learn(data, features, y, depth) #get model
        error, errlog = calcError(h,data,features,y) #find error of model
        alpha = .5*math.log((1 - error)/error) #calc weight
        #TODO: what happens if error = 0 or 1?, comments in slides say only if error<0.5

        newData = copy.deepcopy(data)
        #calculations for D[l+1]
        i = 0 #example index
        for index,ex in data.iterrows():
        # for err in errlog: 
            # Dex = 0
            if (errlog[i]): #guessed wrong
                # Dex = D[l][i]*math.exp(alpha)
                newData['D'][index] = ex['D']*math.exp(alpha)
            else: #guessed right
                # Dex = D[l][i]*math.exp(-alpha)
                newData['D'][index] = ex['D']*math.exp(-alpha)
            # print('prevD: {}, newD: {}, alpha: {}, err: {}'.format(ex['D'],newData['D'][index],alpha, errlog[i]))
            # Dl.append(Dex) #save D[l+1][ex]
            i += 1 #increment for next loop
        data = newData

        #normalize D
        print('Sum of D before: {0}'.format(data.sum(axis=0)['D']))
        Norm = data.sum(axis=0)['D'] #TODO: make sure this is how you normalize
        data['D'] = data['D']/Norm
        print('Sum of D after: {0}'.format(data.sum(axis=0)['D']))
        # D.append(Dl) #save D[l+1]
        H.append(h)  #save model for learner
        Alpha.append(alpha) #save vote weight for learner

    return H, Alpha


# #weak learner (DT with depth and weighted examples)
# def learn(data,features,y,depth):
#     # h = []

#     # N = len(data) # number of examples
#     # # print('# of examples: {0}'.format(N))

#     # #initialize weights D, with uniform distribution initially
#     # # D = [] #weights for each example, for each learner
#     # # D.append(np.ones(N)/N)
#     # data['D'] = np.ones(N)/N

#     c0 = len(data.loc[data[y] == -1])
#     c1 = len(data.loc[data[y] == 1])
#     sumD1 = sum(data.loc[data[y] == 1]['D']) #0
#     sumD0 = sum(data.loc[data[y] == -1]['D']) # 0
#     # for i in range(len(data)):
#     #     if data[i][y] == 1:
#     #         sumD1 += data[i]['D']
#     #     else:
#     #         sumD0 += data[i]['D']
    

#     p1 = (sumD1/(sumD1+sumD0))				# probabilty that y=1
#     p0 = (sumD0/(sumD1+sumD0))				# probabilty that y=0

#     tree = {}
#     tree['root'] = {}
#     tree['root']['data'] = data
#     tree['root']['f=0'] = c0
#     tree['root']['f=1'] = c1
#     tree['root']['prob'] = 1
#     tree['root']['U'] = 1 - p1**2 - p0**2				#note: want to confirm this.
#     tree['root']['continue'] = True
#     tree['root']['split_on'] = None
#     tree['root']['p1'] = p1
#     tree['root']['p0'] = p0


#     for i in range(depth):
#         temp = list(itertools.product([1,0], repeat=i+1))

#         if len(temp[0]) == 1:
#             temp = [str(i[0]) for i in temp]
#         else:
#             temp = ['-'.join(map(str, i)) for i in temp]
#         for ii in temp:
#             tree[ii] = {}
#             tree[ii]['data'] = None
#             tree[ii]['f=0'] = None
#             tree[ii]['f=1'] = None
#             tree[ii]['prob'] = None
#             tree[ii]['U'] = None
#             tree[ii]['continue'] = None
#             tree[ii]['split_on'] = None
#             tree[ii]['p1'] = None
#             tree[ii]['p0'] = None


#     tree = HF.build_tree(tree, features, y, depth, mode='adaboost')

#     # for i in range(1,depth+1):
#     #     if (len(features)) > 0:
#     #         tree, features = HF.find_best_feature(tree, layer=i, features=features, y=y, mode='adaboost')

#     tree = {i:j for i,j in tree.items() if j != {}}	# removing empty keys


#     return tree


#calculate the weighted error, returns a log of the results too
#for the log, 0 if guess is correct, 1 if wrong
def calcError(h,data,features,y):
    errlog = []
    error = 0

    # i = 0 #example index
    for index,ex in data.iterrows():
        # print('\n\n')
        # print(ex)
        if (useModel(h,ex) == ex[y]):
            errlog.append(0) 
        else:
            errlog.append(1)
            error += ex['D']
        # i += 1
    print('Total number of errors: {0}'.format(error))
    return error, errlog


#using a single learned model of adaboost
#returns 1 if poisonous
def useModel(tree,ex):
    #start with root
    node = 'root'
    feat = tree[node]['split_on'] #what feature to split on
    node = str(int(ex[feat]))
    # print('feat: {}, node: {}'.format(feat, node))
    # print(tree.keys())
    if tree[node]['split_on'] is None:
        if tree[node]['p1'] >= tree[node]['p0']:
            return 1
        else:
            return -1

    #loop until done
    while(True):
        feat = tree[node]['split_on']
        # print('feat: {}'.format(feat))
        newNode = node + '-' + str(int(ex[feat])) #set next node
        #check if newNode exists
        # if newNode not in tree.keys():
        if tree[node]['split_on'] is None:
            if tree[node]['p1'] >= tree[node]['p0']:
                return 1
            else:
                return -1
        node = newNode


#using the adaboost model
#returns 1 if poisonous
def toEatOrNotToEat(learners,weight,features):
    isPoisonous = -1
    weighted_guess = 0
    i = 0
    for tree in learners:
        guess = useModel(tree,features)
        weighted_guess += guess*weight[i]
        i += 1
        
    if weighted_guess >= 0:
        isPoisonous = 1
    else:
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
# print(data['cap-shape_c'][1])
# print(features)
# print(y)

H_all = [] #all the models
Alpha_all = [] #all the vote weights

# Part C
depth = 1
learners = [5] #[1,2,5,10,15] #number of learners to try
for L in learners:
    # H, Alpha = adaboost(S,y,L,1) #run adaboost
    H, Alpha = adaboost(data,features,y,L,depth) #run adaboost
    #save results
    H_all.append(H)
    Alpha_all.append(Alpha)
    # path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part3', 'TEST_L{}D{}.csv' .format(L,depth))
    #try results
    howmanywrong = 0
    for index,ex in data.iterrows():
        guess = toEatOrNotToEat(H,Alpha,ex)
        if guess != ex[y]:
            howmanywrong += 1
    print('{} many wrong'.format(howmanywrong))


# HF.write_out_tree(tree, path_to_outfile)

# Part E
# learners = [1,2,5,10,15] #number of learners to try
# for L in learners:
#     H, Alpha = adaboost(S,y,L) #run adaboost
#     #save results
#     H_all.append(H)
#     Alpha_all.append(Alpha)
