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
	df['class'].replace(0, -1,inplace=True)
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
		#	the layer number is the same as the length of the node name 
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
	children_remove = []
	for node_i in tree.keys():
		split_node = node_i.split('-')
		len_node = len(node.split('-'))
		if len(split_node) > len_node:
			if '-'.join(split_node[0:len_node]) == node:
				children_remove.append(node_i)
				# tree.pop('{}'.format(node_i), None)

	return children_remove


def split_tree(tree, node, features, y, depth):
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

	if node == 'root':
		parent_node = 'root'
	elif len(node.split('-')) == 1:
		parent_node = 'root'
	else:
		parent_node = node.split('-')[:-1]				# finding parent node (e.g. node 1-0-1's parent is node 1-0)
		parent_node = '-'.join(parent_node)				# 	re-joining after split above
	
	if tree[parent_node]['continue'] == True:
		if node == 'root':
			children = ['1', '0']
		else:
			children = [node+'-1', node+'-0']

		sel_B = 0
		for feature in features:
			if feature not in tree[node]['feature_path']:
				temp_data = tree[node]['data']
				data_1 = temp_data.loc[temp_data[feature]==1]
				data_0 = temp_data.loc[temp_data[feature]==0]

				c1_1 = len(data_1.loc[data_1[y]==1])	# counting number of features with y=1
				c1_0 = len(data_1.loc[data_1[y]==-1])	# 	same for y=0

				c0_1 = len(data_0.loc[data_0[y]==1])	# counting number of features with y=1
				c0_0 = len(data_0.loc[data_0[y]==-1])	# 	same for y=0

				p1 = len(data_1)/len(temp_data)
				p0 = len(data_0)/len(temp_data)

				if (c1_1 == 0) and (c1_0 == 0):
					p1_1 = 0
					p1_0 = 0
				else:
					p1_1 = (c1_1/(c1_1+c1_0))				# probabilty that y=1
					p1_0 = (c1_0/(c1_1+c1_0))				# probabilty that y=0

				if (c0_1 == 0) and (c0_0 == 0):
					p0_1 = 0
					p0_0 = 0
				else:
					p0_1 = (c0_1/(c0_1+c0_0))				# probabilty that y=1
					p0_0 = (c0_0/(c0_1+c0_0))				# probabilty that y=0

				U_1 = 1 - (p1_1**2) - (p1_0**2)
				U_0 = 1 - (p0_1**2) - (p0_0**2)
				B = tree[node]['U'] - p1*U_1 - p0*U_0
				if B > sel_B:
					sel_B = B
					tree[children[0]]['f=1'] = c1_1				# count of 1s
					tree[children[0]]['f=0'] = c1_0				# count of 0s
					tree[children[0]]['data'] = data_1			# data
					tree[children[0]]['prob'] = p1	 			# prob. of being on this particular node
					tree[children[0]]['p1'] = p1_1				# prob of y=1
					tree[children[0]]['p0'] = p1_0				# prob of y=0
					tree[children[0]]['U'] = U_1				# Uncertainty
					if len(tree[children[0]]['feature_path']) > 1:
						tree[children[0]]['feature_path'][-1] = feature 	# feature that the data was split on
					else:
						tree[children[0]]['feature_path'].append(feature)

					tree[children[1]]['f=1'] = c0_1				# count of 1s
					tree[children[1]]['f=0'] = c0_0				# count of 0s
					tree[children[1]]['data'] = data_0			# data
					tree[children[1]]['prob'] = p0	 			# prob. of being on this particular node
					tree[children[1]]['p1'] = p0_1				# prob of y=1
					tree[children[1]]['p0'] = p0_0				# prob of y=0
					tree[children[1]]['U'] = U_0				# Uncertainty
					

					if len(tree[children[1]]['feature_path']) > 1:
						tree[children[1]]['feature_path'][-1] = feature 	# feature that the data was split on
					else:
						tree[children[1]]['feature_path'].append(feature)


					if (c1_1 == 0) or (c1_0 == 0):
						all_children = get_children_nodes(tree, node)
						tree[children[0]]['continue'] = False
						print(node)
						print('\t{}' .format(children[0]))
						for child in all_children:
							tree[child]['continue'] = False
					if (c0_1 == 0) or (c0_0 == 0):
						all_children = get_children_nodes(tree, node)
						tree[children[1]]['continue'] = False
						for child in all_children:
							tree[child]['continue'] = False
	return tree

def get_children_nodes(tree, node):
	children = []
	for node_i in tree.keys():
		split_node = node_i.split('-')
		len_node = len(node.split('-'))
		if len(split_node) > len_node:
			if '-'.join(split_node[0:len_node]) == node:
				children.append(node_i)
				# tree.pop('{}'.format(node_i), None)
	return children

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
		B = tree['root']['U'] - tot_gini		# note: want to confirm this.
	return B

def build_tree(tree, features, y, depth):
	tree_out = copy.deepcopy(tree)
	
	for node in tree.keys():
		if len(node.split('-')) == depth:
			break
		tree_out = split_tree(tree_out, node, features, y, depth)
	return tree_out


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
	p1 = []
	p0 = []

	for key in tree.keys():
		node.append(key)
		split.append(tree[key]['split_on'])
		p1.append(tree[key]['p1'])
		p0.append(tree[key]['p0'])

	outdata = pd.DataFrame(
	    {'node': node,
	     'split': split,
	     'p1': p1,
	     'p0': p0
	    })

	outdata.to_csv(path_to_outfile, index=False, na_rep="None")


def predict_with_tree(path_to_tree_csv, data, y=None):

	# --- creating the tree --- 
	tree_csv = pd.read_csv(path_to_tree_csv)
	tree = {}
	# for node in tree_csv['node']:
	for index, row in tree_csv.iterrows():
		node = row['node']
		tree[node] = {}
		tree[node]['split_on'] = row['split']
		tree[node]['p1'] = row['p1']
		tree[node]['p0'] = row['p0']

	# --- running data through tree ---
	tree['root']['data'] = data
	for node in tree.keys():
		data = tree[node]['data']
		if tree[node]['split_on'] != 'None':
			if node == 'root':
				children = ['1', '0']
			else:
				children = [node+'-1', node+'-0']

			for child in children:
				split_val = int(child.split('-')[-1])
				child_data = data.loc[data[tree[node]['split_on']]==split_val]	# where feature is 1
				tree[child]['data'] = child_data



	return tree







