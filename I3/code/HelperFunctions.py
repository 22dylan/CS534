import numpy as np
import pandas as pd
import copy
import itertools
import os

def datareader(path_to_data):
    """ 
    returns data as a numpy array

    input: 
        path_to_data - path to data
    output: 
        data - a dictionary of the data. each key in the 
            dict corresponds to a column heading (e.g. cap-shape_f)
    """

    df = pd.read_csv(path_to_data, dtype='float') #you wanted float datatype
    df = df.drop(['veil-type_p'], axis=1)
    if 'class' in list(df.columns.values):
        df['class'].replace(0, -1,inplace=True)
    return df


def split_tree(tree, node, features, y, depth, mode='DT'):
    """
        splitting tree based on a given feature.

        steps:
            1) loop through all nodes in the specified layer
            2) perform split on data
            3) calculate necessary values (count of y=1, count of y=0, etc.)

        input: 
            tree: dictionary of the tree to split
            feature: feature to perform the split on
            consider_nodes: numpy array of nodes in the layer to split  
            y: feature that is being fit to
        output:
            tree: an updated tree with the additional split
            continue_tf_: indicator of whether to build off children nodes
    """
    continue_tf_1 = True
    continue_tf_0 = True
    if tree[node]['continue'] == True:
        if node == 'root':
            children = ['1', '0']
        else:
            children = [node+'-1', node+'-0']

        sel_B = 0
        sel_feature = features[0]
        for feature in features:
            if feature not in tree[node]['feature_path']:       # check if feature has been used yet. 
                temp_data = tree[node]['data']                  # isolating data split on
                data_1 = temp_data.loc[temp_data[feature]==1]   # split where feature is 1
                data_0 = temp_data.loc[temp_data[feature]==0]   # split where feature is 0

                c1_1 = len(data_1.loc[data_1[y] == 1])  # counting number of features with y=1
                c1_0 = len(data_1.loc[data_1[y] == -1]) #   same for y=-1

                c0_1 = len(data_0.loc[data_0[y] == 1])  # counting number of features with y=1
                c0_0 = len(data_0.loc[data_0[y] == -1]) #   same for y=-1

                p1 = len(data_1)/len(temp_data)
                p0 = len(data_0)/len(temp_data)
                if mode == 'adaboost':
                    sum1 = (data_1.sum()['D']) #0
                    sum0 = (data_0.sum()['D'])
                    sum_temp = (temp_data.sum()['D']) # 0
                    p1 = (sum1/(sum_temp))                # probabilty that y=1
                    p0 = (sum0/(sum_temp))                # probabilty that y=0

                if (c1_1 == 0) and (c1_0 == 0):
                    p1_1 = 0
                    p1_0 = 0
                elif mode=='adaboost':
                    sumD1 = (data_1.loc[data_1[y] == 1].sum()['D']) #0
                    sumD0 = (data_1.loc[data_1[y] == -1].sum()['D']) # 0
                    p1_1 = (sumD1/(sumD1+sumD0))                # probabilty that y=1
                    p1_0 = (sumD0/(sumD1+sumD0))                # probabilty that y=0
                else:
                    p1_1 = (c1_1/(c1_1+c1_0))               # probabilty that y=1
                    p1_0 = (c1_0/(c1_1+c1_0))               # probabilty that y=0


                if (c0_1 == 0) and (c0_0 == 0):
                    p0_1 = 0
                    p0_0 = 0
                elif mode=='adaboost':
                    sumD1 = (data_0.loc[data_0[y] == 1].sum()['D'])     #0
                    sumD0 = (data_0.loc[data_0[y] == -1].sum()['D'])    # 0
                    p0_1 = (sumD1/(sumD1+sumD0))                # probabilty that y=1
                    p0_0 = (sumD0/(sumD1+sumD0))                # probabilty that y=0
                else:
                    p0_1 = (c0_1/(c0_1+c0_0))               # probabilty that y=1
                    p0_0 = (c0_0/(c0_1+c0_0))               # probabilty that y=0

                U_1 = 1 - (p1_1**2) - (p1_0**2)
                U_0 = 1 - (p0_1**2) - (p0_0**2)
                B = tree[node]['U'] - p1*U_1 - p0*U_0
                if B > sel_B:
                    continue_tf_1 = True
                    continue_tf_0 = True
                    
                    sel_B = B
                    sel_feature = feature

                    tree[children[0]]['f=1'] = c1_1             # count of 1s
                    tree[children[0]]['f=0'] = c1_0             # count of 0s
                    tree[children[0]]['data'] = data_1          # data
                    tree[children[0]]['prob'] = p1              # prob. of being on this particular node
                    tree[children[0]]['p1'] = p1_1              # prob of y=1
                    tree[children[0]]['p0'] = p1_0              # prob of y=0
                    tree[children[0]]['U'] = U_1                # Uncertainty
                    # tree[children[0]]['split_on'] = sel_feature # feature that data was split on

                    tree[children[1]]['f=1'] = c0_1             # count of 1s
                    tree[children[1]]['f=0'] = c0_0             # count of 0s
                    tree[children[1]]['data'] = data_0          # data
                    tree[children[1]]['prob'] = p0              # prob. of being on this particular node
                    tree[children[1]]['p1'] = p0_1              # prob of y=1
                    tree[children[1]]['p0'] = p0_0              # prob of y=0
                    tree[children[1]]['U'] = U_0                # Uncertainty
                    # tree[children[1]]['split_on'] = sel_feature # feature that data was split on
                    
                    if (c1_1 == 0) or (c1_0 == 0):
                        continue_tf_1 = False
                        # tree[children[0]]['split_on'] == None

                    if (c0_1 == 0) or (c0_0 == 0):
                        continue_tf_0 = False
                        
        if node == 'root':
            tree[node]['feature_path'].append(sel_feature)
            tree[node]['split_on'] = sel_feature
        else:
            if len(node.split('-')) == 1:
                parent_node = 'root'
            else:
                parent_node = node.split('-')[:-1]              # finding parent node (e.g. node 1-0-1's parent is node 1-0)
                parent_node = '-'.join(parent_node)             #   re-joining after split above
            
            tree[node]['feature_path'] = tree[parent_node]['feature_path'][:]
            tree[node]['feature_path'].append(sel_feature)
            tree[node]['split_on'] = sel_feature

    return tree, continue_tf_1, continue_tf_0

def get_children_nodes(tree, node):
    children = []
    for node_i in tree.keys():
        split_node = node_i.split('-')
        len_node = len(node.split('-'))
        if len(split_node) > len_node:
            if '-'.join(split_node[0:len_node]) == node:
                children.append(node_i)
    return children

def populate_tree(tree, features, y, depth, mode='DT'):
    nodes = list(tree.keys())
    for node in nodes:
        if (len(node.split('-')) == depth) and (node!='root'):
            continue
        tree, continue_tf_1, continue_tf_0 = split_tree(tree, node, features, y, depth, mode)

        if continue_tf_0 == False:
            tree = remove_children(tree, node, '0')
        if continue_tf_1 == False:
            tree = remove_children(tree, node, '1')
    return tree

def remove_children(tree, node, child):
    if node == 'root':
        child = child
    else:
        child = node + '-' + child
    all_children = get_children_nodes(tree, child)
    for child in all_children:
        tree[child]['continue'] = False
    return tree

def build_empty_tree(data, features, y, depth, mode):
    """
    creating tree by predefining all possible paths 
      -example of nomenclature:
                    root
                   /    \
                 1        0
               /   \    /   \
              11   10  01    00
        etc.etc.etc.etc.etc.etc.etc.etc.
                  etcetera
                    etc.
    """

    c0 = len(data.loc[data[y] == -1])
    c1 = len(data.loc[data[y] == 1])

    if mode == 'adaboost':
        sumD1 = sum(data.loc[data[y] == 1]['D']) #0
        sumD0 = sum(data.loc[data[y] == -1]['D']) # 0
        p1 = (sumD1/(sumD1+sumD0))              # probabilty that y=1
        p0 = (sumD0/(sumD1+sumD0))              # probabilty that y=0
    else:
        p1 = (c1/(c1+c0))             # probabilty that y=1
        p0 = (c0/(c1+c0))             # probabilty that y=0

    tree = {}
    tree['root'] = {}
    tree['root']['data'] = data
    tree['root']['f=0'] = c0
    tree['root']['f=1'] = c1
    tree['root']['prob'] = 1
    tree['root']['U'] = 1 - p1**2 - p0**2
    tree['root']['continue'] = True
    tree['root']['split_on'] = None
    tree['root']['p1'] = p1
    tree['root']['p0'] = p0
    tree['root']['feature_path'] = []

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
            tree[ii]['continue'] = True
            tree[ii]['split_on'] = None
            tree[ii]['p1'] = None
            tree[ii]['p0'] = None
            tree[ii]['feature_path'] = []

    return tree

def print_tree(tree):
    for key in tree.keys():
      print('key: {}' .format(key))
      print('f=0: {}' .format(tree[key]['f=0']))
      print('f=1: {}' .format(tree[key]['f=1']))
      print('prob: {}' .format(tree[key]['prob']))
      print('U: {}' .format(tree[key]['U']))
      print('continue: {}' .format(tree[key]['continue']))
      print('split_on: {}' .format(tree[key]['split_on']))
      print('p1: {}' .format(tree[key]['p1']))
      print('p0: {}' .format(tree[key]['p0']))
      print('feature_path: {}' .format(tree[key]['feature_path']))
      print('____________________________________________')


def learn(data,features,y,depth, mode='DT', view_tree=False):
    tree = build_empty_tree(data, features, y, depth, mode)
    tree = populate_tree(tree, features, y, depth, mode=mode)
    tree = {i:j for i,j in tree.items() if tree[i]['p1'] != None}   # removing empty keys

    if view_tree == True:
        print_tree(tree)

    return tree

def write_out_tree(tree, path_to_outfile):
    # writing tree to csv
    node = []
    split = []
    p1 = []
    p0 = []
    feature_path = []

    for key in tree.keys():
        if key == 'root':
            path = None
        elif (key == '1') or (key == '0'):
            path = tree['root']['feature_path']
        else:
            parent = key.split('-')[0:len(key.split('-'))-1]
            parent = '-'.join(parent)
            path = tree[parent]['feature_path']

        if len(tree[key]['feature_path']) >= 1:
            if (tree[key]['p1'] != 0) and (tree[key]['p0']!=0):
                split.append(tree[key]['feature_path'][-1])
            else:
                split.append(None)
        else:
            split.append(None)
        
        node.append(key)
        p1.append(tree[key]['p1'])
        p0.append(tree[key]['p0'])
        feature_path.append(path)


    outdata = pd.DataFrame(
        {'node': node,
         'split': split,
         'p1': p1,
         'p0': p0, 
         'feature_path': feature_path
        })

    outdata.to_csv(path_to_outfile, index=False, na_rep="None")

def build_tree_from_csv(path_to_csv, data, y=None):
    # --- creating the tree ---
    tree_csv = pd.read_csv(path_to_csv)
    tree_csv.replace('None', None, inplace=True)
    tree = {}
    for index, row in tree_csv.iterrows():
        node = row['node']
        tree[node] = {}
        tree[node]['split_on'] = row['split']
        tree[node]['p1'] = row['p1']
        tree[node]['p0'] = row['p0']
        tree[node]['feature_path'] = row['feature_path']

    # # --- running data through tree ---
    # tree['root']['data'] = data
    # for node in tree.keys():
    #     data = tree[node]['data']
    #     if tree[node]['split_on'] != 'None':
    #         if node == 'root':
    #             children = ['1', '0']
    #         else:
    #             children = [node+'-1', node+'-0']
    #         for child in children:
    #             split_val = int(child.split('-')[-1])
    #             child_data = data.loc[data[tree[node]['split_on']]==split_val]  # where feature is 1
    #             tree[child]['data'] = child_data

    return tree

def calc_error(path_to_csv, data, y=None, mode='DT'):
    tree = build_tree_from_csv(path_to_csv, data)
    error_count = 0
    errlog = []
    y_pred_tot = []
    for index,ex in data.iterrows():
        y_pred = useTree(tree, ex)
        y_pred_tot.append(y_pred)

        if y != None:
            if (y_pred == ex[y]):
                errlog.append(0) 
            else:
                errlog.append(1)
                if mode=='adaboost':
                    error_count += ex['D']
                else:
                    error_count+=1
        # i += 1
    return error_count, y_pred_tot

#using a single learned tree
#returns 1 if poisonous
def useTree(tree,ex):
    #start with root
    node = 'root'
    feat = tree[node]['split_on'] #what feature to split on
    node = str(int(ex[feat]))
    # print('feat: {}, node: {}'.format(feat, node))
    if (tree[node]['split_on'] is None) or (tree[node]['split_on'] == "None"):
        if tree[node]['p1'] >= tree[node]['p0']:
            return 1
        else:
            return -1

    #loop until done
    while(True):
        feat = tree[node]['split_on']

        if (tree[node]['split_on'] is None) or (tree[node]['split_on'] == "None"):
            if tree[node]['p1'] >= tree[node]['p0']:
                return 1
            else:
                return -1
                
        newNode = node + '-' + str(int(ex[feat])) #set next node
        if newNode not in tree.keys():
            if tree[node]['p1'] >= tree[node]['p0']:
                return 1
            else:
                return -1

        node = newNode


