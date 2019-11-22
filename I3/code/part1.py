import os 
import numpy as np
import HelperFunctions as HF
import pandas as pd
import itertools


# -- all data --- 
path_to_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
data0 = HF.datareader(path_to_data)
features= list(data0.columns[:-1])
y = data0.columns[-1]

# # --- manageable test data --- 
# data0 = np.array([[0,0,0, 1],
# 				[0,0,1, 0],
# 				[0,1,0, 0],
# 				[0,1,1, 1],
# 				[1,0,0, 0],
# 				[1,0,1, 0],
# 				[1,1,0, 1],
# 				[1,1,1, 1]])

# column_names = ['x1', 'x2', 'x3', 'f']
# data0 = pd.DataFrame(data0)
# data0.columns = column_names
# features = column_names[:-1]
# y = column_names[-1]



# --- Part 1a ---
# create a tree with a maximum depth of 2
depth = 5

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
c0 = len(data0.loc[data0[y] == 0])
c1 = len(data0.loc[data0[y] == 1])

tree = {}
tree['root'] = {}
tree['root']['data'] = data0
tree['root']['f=0'] = c0
tree['root']['f=1'] = c1
tree['root']['prob'] = 1
tree['root']['U'] = 1 - (c1/(c1+c0))**2 - (c0/(c1+c0))**2				#note: want to confirm this.
tree['root']['continue'] = True
tree['root']['split_on'] = None

for i in range(depth):
	temp = list(itertools.product([1,0], repeat=i+1))

	if len(temp[0]) == 1:
		temp = [str(i[0]) for i in temp]
	else:
		temp = ['-'.join(map(str, i)) for i in temp]
	for ii in temp:
		tree[ii] = {}

for i in range(1,depth+1):
	if (len(features)) > 0:
		tree, features = HF.find_best_feature(tree, layer=i, features=features, y=y)

tree = {i:j for i,j in tree.items() if j != {}}	# removing empty keys

path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part1', 'part1a_D{}.csv' .format(depth))
HF.write_out_tree(tree, path_to_outfile)






# # --- Part 1c ---
# # create a tree with a maximum depth ranging from 1-8
# depth_vals = [1, 2, 3, 4, 5, 6, 7, 8]

# for depth in depth_vals:
# 	print(depth)

# 	# re-reading data for a fresh start
# 	path_to_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
# 	data0 = HF.datareader(path_to_data)
# 	features= list(data0.columns[:-1])
# 	y = data0.columns[-1]

# 	c0 = len(data0.loc[data0[y] == 0])
# 	c1 = len(data0.loc[data0[y] == 1])

# 	tree = {}
# 	tree['root'] = {}
# 	tree['root']['data'] = data0
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

# 	for i in range(1,depth+1):
# 		if (len(features)) > 0:
# 			tree, features = HF.find_best_feature(tree, layer=i, features=features, y=y)

# 	tree = {i:j for i,j in tree.items() if j != {}}	# removing empty keys

# 	path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part1', 'part1c_D{}.csv' .format(depth))
# 	HF.write_out_tree(tree, path_to_outfile)



