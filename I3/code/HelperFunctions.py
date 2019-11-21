import numpy as np
import pandas as pd
import copy

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
	df.drop(['veil-type_p'], axis=1)
	# data = df.to_dict(orient='list')
	# for key in data.keys():
	# 	data[key] = np.array(data[key])

	return df

# -------------------------- OLD ------------------------------------

# def create_split(data, feature, y):
# 	"""
# 		creates a split in the decsion tree as such:

# 				[c_p, c_m]
# 			  	/		\
# 		    pl /      pr \
# 	 	      /           \
# 		[cl_p, cl_m]  	[cr_p, cr_m]

# 	"""

# 	# splitting data based on passed in feature
# 	data_t = data.loc[data[feature] == 1]
# 	data_f = data.loc[data[feature] == 0]


# 	# counting number of true/false variables at each node
# 	c_p = len(data.loc[data[y] == 1])
# 	c_m = len(data.loc[data[y] == 0])
# 	cl_p = len(data_t.loc[data[y] == 1])
# 	cl_m = len(data_t.loc[data[y] == 0])
# 	cr_p = len(data_f.loc[data[y] == 1])
# 	cr_m = len(data_f.loc[data[y] == 0])
# 	if 0 in [cl_p, cl_m, cr_p, cr_m, c_p, c_m]:
# 		B = -999
# 		data_t = None
# 		data_f = None
# 	else:
# 		# counting probability of left and right nodes
# 		pl = (cl_p + cl_m)/(c_p + c_m)
# 		pr = (cr_p + cr_m)/(c_p + c_m)

# 		# computing gini indicies
# 		U_al = 1 - (cl_p/(cl_p + cl_m))**2 - (cl_m/(cl_p + cl_m))**2
# 		U_ar = 1 - (cr_p/(cr_p + cr_m))**2 - (cr_m/(cr_p + cr_m))**2
# 		U_a = 1 - (c_p/(c_p+c_m))**2 - (c_m/(c_p+c_m))**2				#note: want to confirm this.

# 		# computing benefit
# 		B = U_a - pl*U_al - pr*U_ar

# 	return data_t, data_f, B

# def find_best_feature(data, features, y):
# 	"""
# 	finds the best feature to create a split on 
# 		based on the benefit of each split
# 	"""
# 	features_copy = features.copy()
# 	sel_feature = features[0]
# 	sel_B = 0
# 	for feature in features:
# 		data_t, data_f, B = create_split(data, feature, y)

# 		if B >= sel_B:
# 			sel_feature = feature
# 			sel_B = B
# 			sel_data_t = data_t
# 			sel_data_f = data_f

# 	features_copy.remove(sel_feature)
# 	return sel_data_t, sel_data_f, features_copy, sel_feature

# -------------------------- OLD ------------------------------------




def find_considered_nodes(tree, layer):
	# first finding all nodes in the specified layer
	all_nodes = np.array(list(tree.keys()))
	node_level = np.zeros(len(all_nodes))
	for i in range(len(all_nodes)):
		node_level[i] = np.sum(c.isdigit() for c in all_nodes[i])

	node_level = np.array(node_level)
	consider_nodes_idx = np.array(node_level==layer)
	consider_nodes = all_nodes[consider_nodes_idx]	# isolate nodes
	return consider_nodes

def test_split(tree, feature, consider_nodes, y):
	"""
		some brief description that makes sense
	"""

	for node in consider_nodes:			# looping through all nodes in specified layer
		if len(node.split('-')) == 1:	# if the first layer, use the root data
			parent_node = 'root'
		else:							# otherwise, use the data stored in the parent node 
			parent_node = node.split('-')[:-1]	# (e.g. node 1-0-1 uses data in node 1-0)
			parent_node = '-'.join(parent_node)

		if tree[parent_node]['continue'] == True:	# if node can be built off of
			temp_data = tree[parent_node]['data']	# gettting data from the parent node
			
			""" isolating which value to test on 
				e.g. if at node 1-0-1 and testing on feature 'x1', 
					test_feature_value isolates all data points in 
					temp_data where x1 = 1 (last value in 1-0-1)
			"""
			test_feature_value = int(node[-1])	
			data_1 = temp_data.loc[temp_data[feature]==test_feature_value]

			c1 = len(data_1.loc[data_1[y]==1])	# counting number of features with y=1
			c0 = len(data_1.loc[data_1[y]==0])	# 	same for y=0
			p1 = (c1/(c1+c0))					# probabilty of y=1
			p0 = (c0/(c1+c0))					# probabilty of y=0

			# populating tree
			tree[node]['f=1'] = c1				# count of 1s
			tree[node]['f=0'] = c0				# count of 0s
			tree[node]['data'] = data_1			# storing data
			tree[node]['prob'] = len(data_1)/len(tree['root']['data'])	# storing prob. of being on this particular node
			tree[node]['p1'] = p1				# prob of y=1
			tree[node]['p0'] = p0				# prob of y=0
			tree[node]['U'] = 1 - p1**2 - p0**2	# calculating U(node)
			tree[node]['split_on'] = feature 	# feature that the data was split on
			
			# adding a check if the node can be built off of
			tree[node]['continue'] = True 		# initilizing with true
			if (p1==1.) or (p0==1.):			# testing of data is already separated completely. 
				tree[node]['continue'] = False
		
		else:
			tree.pop('{}'.format(node), None)


	return tree

def gini_calc(tree, nodes):

	tot_gini = 0
	for node in nodes:
		u = tree[node]['U'] 
		p = tree[node]['prob']
		tot_gini += p*u
	B = tree['root']['U'] - tot_gini
	return B, tree

def find_best_feature2(tree, layer, features, y):
	features_copy = features.copy()

	consider_nodes = find_considered_nodes(tree, layer)

	sel_feature = features[0]
	sel_B = 0
	for feature in features:
		tree_temp = test_split(tree, feature, consider_nodes, y)
		nodes_temp = find_considered_nodes(tree, layer)
		B, tree_temp = gini_calc(tree_temp, nodes_temp)

		if B >= sel_B:
			sel_feature = feature
			sel_B = B
			tree_out = copy.deepcopy(tree_temp)

	features_copy.remove(sel_feature)
	tree = copy.deepcopy(tree_temp)

	return tree_out, features_copy
