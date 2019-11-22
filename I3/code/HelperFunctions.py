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
	df = df.drop(['veil-type_p'], axis=1)
	# data = df.to_dict(orient='list')
	# for key in data.keys():
	# 	data[key] = np.array(data[key])

	return df

def find_considered_nodes(tree, layer):
	"""
	finding names of all nodes in the specified layer
		for example: 
			-layer=1: consider_nodes = ['1', '0']
			-layer=2: consider_nodes = ['1-1', '1-0', '0-1', '0-0']
			-etc.
	steps:
		1) finds all possible nodes in the layer
		2) check that parent node exists in tree

	input:
		tree:  a dictionary of the tree under consideration
		layer: the layer to consider
	output:
		consider_nodes: a numpy array of the nodes in the layer
	"""
	all_nodes = np.array(list(tree.keys()))		# all nodes in tree
	node_level = np.zeros(len(all_nodes))		# pre-allocating some space
	for i in range(len(all_nodes)):				# looping through all nodes
		# figuring out the layer that each node belongs to.
		#	note that the layer is the same as the length of hte node name 
		#	e.g. [1-0] is in layer 2, [1-0-0] is in layer 3
		node_level[i] = np.sum(c.isdigit() for c in all_nodes[i])

	consider_nodes_idx = np.array(node_level==layer)	# finding index of nodes in layer
	consider_nodes = all_nodes[consider_nodes_idx]		# isolate those nodes

	# --- check if parent exists in tree --- 
	if layer > 1:		# assuming that the root exists
		consider_nodes_copy = np.copy(consider_nodes)		# making a copy to loop through as the original maybe modified
		for node in consider_nodes_copy:					# looping through nodes in layer
			parent_node = node.split('-')[:-1]				# finding parent node (e.g. node 1-0-1's parent is node 1-0)
			parent_node = '-'.join(parent_node)				# 	re-joining after split above

			if parent_node not in all_nodes:				# if the parent node doesn't exist
				idx = np.argwhere(consider_nodes==node)		# find index in all nodes
				consider_nodes = np.delete(consider_nodes, idx)	# and delete that bad bessy
		
	return consider_nodes

def remove_children_nodes(tree, node):
	for node_i in tree.keys():
		split_node = node_i.split('-')
		len_node = len(node.split('-'))
		if len(split_node) > len_node:
			if '-'.join(split_node[0:len_node]) == node:
				tree.pop('{}'.format(node), None)
	return tree



def split_tree(tree, feature, consider_nodes, y):
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

			if (c1==0) and (c0==0):	 		# if there's no features at split
				p0 = 0
				p1 = 0
			else:
				p1 = (c1/(c1+c0))				# probabilty that y=1
				p0 = (c0/(c1+c0))				# probabilty that y=0

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
			if (p1==0.) or (p0==0.):			# testing of data is already separated completely. 
				tree[node]['continue'] = False
		
		else:					# if node can't be built off of, remove from tree
			tree.pop('{}'.format(node), None)
			# tree = remove_children_nodes(tree, node)	# note: drs, stopping here
	return tree

def benefit_calc(tree, nodes, type_split='gini'):

	"""
	performs benefit of split calculation

	input: 
		tree:  dictionary of tree
		nodes: nodes to calculate benefit on
		type_split: uncertainty calculation, initialized with gini. 
	output:
		B: benefit of split
	"""
	if type_split == 'gini':
		tot_gini = 0
		for node in nodes:
			u = tree[node]['U'] 
			p = tree[node]['prob']
			tot_gini += p*u
		B = tree['root']['U'] - tot_gini
	return B

def find_best_feature(tree, layer, features, y):

	"""
	finds best feature to perform split on for the given layer.
	input:
		tree: dictionary of tree
		layer: layer to consider
		features: list of features to test on
		y: feature that is being fit to
	output:
		tree_out: dictionary of updated tree
		features_out: copy of features with the selected feature removed.
			e.g.
				features_in = [x1, x2, x3]
				selected_feature = [x2]
				features_out = [x1, x3]
	"""

	features_out = features.copy()	# copying the features; will be modified in loop
	tree_out = copy.deepcopy(tree)	# initializing a copy of the tree
	consider_nodes = find_considered_nodes(tree, layer)	# finding all nodes in the layer

	sel_feature = features[0]		# arbitrarily selecting an optimal feature
	sel_B = 0						# initializing benefit of split to 0
	for feature in features:		# looping through features to test split on
		tree_temp = split_tree(tree, feature, consider_nodes, y)	# testing the split on feature
		nodes_temp = find_considered_nodes(tree, layer)	# getting updated list of nodes in tree (in case a node was removed during split)
		B = benefit_calc(tree_temp, nodes_temp, 'gini')	# calculating benefit of split

		if B >= sel_B:				# if the benefit is better than previous best
			sel_feature = feature 	# update the selected feature to split on
			sel_B = B 				# update the selected benefit
			tree_out = copy.deepcopy(tree_temp) 	# copy the tree

	features_out.remove(sel_feature)	# remove the selected feature from the features list

	return tree_out, features_out


def write_out_tree(tree, path_to_outfile):
	# writing out tree
	node = []
	split = []
	for key in tree.keys():
		node.append(key)
		split.append(tree[key]['split_on'])

	outdata = pd.DataFrame(
	    {'node': node,
	     'split': split
	    })
	outdata.to_csv(path_to_outfile, index=False, na_rep="None")