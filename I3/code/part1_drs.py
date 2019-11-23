import os 
import numpy as np
import HelperFunctions_drs as HF
import pandas as pd
import itertools


# -- all data --- 
path_to_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
data = HF.datareader(path_to_data)
features= list(data.columns[:-1])
y = data.columns[-1]

# # --- manageable test data --- 
# data = np.array([[0,0,0, 1],
# 				[0,0,1, 0],
# 				[0,1,0, 0],
# 				[0,1,1, 1],
# 				[1,0,0, 0],
# 				[1,0,1, 0],
# 				[1,1,0, 1],
# 				[1,1,1, 1]])

# column_names = ['x1', 'x2', 'x3', 'f']
# data = pd.DataFrame(data)
# data.columns = column_names
# features = column_names[:-1]
# y = column_names[-1]



# --- Part 1a ---
""" create a tree with a maximum depth of 2 """
depth = 4

"""  
creating tree by predefining all possible paths 
	-example of nomenclature:
				  root
	             /    \
			   1	    0
		     /   \    /   \
		    11   10  01    00
	  etc.etc.etc.etc.etc.etc.etc.etc.
	  		    etcetera
	  		      etc.
"""
c0 = len(data.loc[data[y] == -1])
c1 = len(data.loc[data[y] == 1])
p1 = (c1/(c1+c0))				# probabilty that y=1
p0 = (c0/(c1+c0))				# probabilty that y=0

tree = {}
tree['root'] = {}
tree['root']['data'] = data
tree['root']['f=0'] = c0
tree['root']['f=1'] = c1
tree['root']['prob'] = 1
tree['root']['U'] = 1 - p1**2 - p0**2				#note: want to confirm this.
tree['root']['continue'] = True
tree['root']['split_on'] = []
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



tree = HF.build_tree(tree, features, y, depth)

# for key in tree.keys():
# 	print(key)
# 	print(tree[key]['f=0'])
# 	print(tree[key]['f=1'])
# 	print(tree[key]['prob'])
# 	print(tree[key]['U'])
# 	print(tree[key]['continue'])
# 	print(tree[key]['split_on'])
# 	print(tree[key]['p1'])
# 	print(tree[key]['p0'])
# 	print(tree[key]['feature_path'])
# 	print('____________________________________________')


# for i in range(1,depth+1):
# 	if (len(features)) > 0:
# 		tree, features = HF.find_best_feature(tree, layer=i, features=features, y=y)

# tree = {i:j for i,j in tree.items() if j != {}}	# removing empty keys

# path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part1', 'TEST_D{}.csv' .format(depth))
# HF.write_out_tree(tree, path_to_outfile)



# # --- part 1b ---
# path_to_train_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
# path_to_val_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_val.csv')
# path_to_tree = os.path.join(os.getcwd(), '..', 'output', 'part1', 'part1a_D2.csv')
# data = HF.datareader(path_to_train_data)


# # --- manageable test data --- 
# path_to_tree = os.path.join(os.getcwd(), '..', 'output', 'part1', 'TEST_D2.csv')
# data = np.array([[0,0,0, 1],
# 				[0,0,1, 0],
# 				[0,1,0, 0],
# 				[0,1,1, 1],
# 				[1,0,0, 0],
# 				[1,0,1, 0],
# 				[1,1,0, 1],
# 				[1,1,1, 1]])

# column_names = ['x1', 'x2', 'x3', 'f']
# data = pd.DataFrame(data)
# data.columns = column_names
# features = column_names[:-1]
# y = column_names[-1]

# tree = HF.predict_with_tree(path_to_tree, data)


# for key in tree.keys():
# 	print(key)
# 	print(tree[key])
# 	print()


# # --- Part 1c ---
# """ create a tree with a maximum depth ranging from 1-8 """
# depth_vals = [1, 2, 3, 4, 5, 6, 7, 8]

# for depth in depth_vals:
# 	print(depth)

# 	# re-reading data for a fresh start
# 	path_to_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
# 	data = HF.datareader(path_to_data)
# 	features= list(data.columns[:-1])
# 	y = data.columns[-1]

# 	c0 = len(data.loc[data[y] == 0])
# 	c1 = len(data.loc[data[y] == 1])

# 	tree = {}
# 	tree['root'] = {}
# 	tree['root']['data'] = data
# 	tree['root']['f=0'] = c0
# 	tree['root']['f=1'] = c1
# 	tree['root']['prob'] = 1
# 	tree['root']['U'] = 1 - (c1/(c1+c0))**2 - (c0/(c1+c0))**2				#note: want to confirm this.
# 	tree['root']['continue'] = True
# 	tree['root']['split_on'] = None

# 	for i in range(depth):
# 		temp = list(itertools.product([1,0], repeat=i+1))

# 		if len(temp[0]) == 1:
# 			temp = [str(i[0]) for i in temp]
# 		else:
# 			temp = ['-'.join(map(str, i)) for i in temp]
# 		for ii in temp:
# 			tree[ii] = {}
# 			tree[ii]['data'] = None
# 			tree[ii]['f=0'] = None
# 			tree[ii]['f=1'] = None
# 			tree[ii]['prob'] = None
# 			tree[ii]['U'] = None
# 			tree[ii]['continue'] = None
# 			tree[ii]['split_on'] = None
# 			tree[ii]['p1'] = None
# 			tree[ii]['p0'] = None

# 	for i in range(1,depth+1):
# 		if (len(features)) > 0:
# 			tree, features = HF.find_best_feature(tree, layer=i, features=features, y=y)

# 	tree = {i:j for i,j in tree.items() if j != {}}	# removing empty keys

# 	path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part1', 'part1c_D{}.csv' .format(depth))
# 	HF.write_out_tree(tree, path_to_outfile)