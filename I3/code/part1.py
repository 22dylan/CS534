import os 
import HelperFunctions as HF


# --- Part 1a ---
""" create a tree with a maximum depth of 2 """

# reading in data
path_to_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
data = HF.datareader(path_to_data)
features= list(data.columns[:-1])
y = data.columns[-1]

# creating and saving tree
depth = 2
tree = HF.learn(data,features,y,depth, mode='DT', view_tree=False)
path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part1', 'part1a_D{}.csv' .format(depth))
HF.write_out_tree(tree, path_to_outfile)


# --- part 1b ---
""" computing training and validation accuracy """
path_to_train_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
path_to_val_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_val.csv')
path_to_tree = os.path.join(os.getcwd(), '..', 'output', 'part1', 'part1a_D{}.csv' .format(depth))

data_trn = HF.datareader(path_to_train_data)
data_val = HF.datareader(path_to_val_data)
y = data_trn.columns[-1]
error_count_trn, _ = HF.calc_error(path_to_tree, data_trn, y)
error_count_val, _ = HF.calc_error(path_to_tree, data_val, y)

acc_trn = 1 - (error_count_trn/len(data_trn))
acc_val = 1 - (error_count_val/len(data_val))
print('Depth {} Training Accuracy: {}' .format(depth, acc_trn))
print('Depth {} Validation Accuracy: {}' .format(depth, acc_val))



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

# 	tree = HF.learn(data, features, y, depth, mode='DT', view_tree=False)
# 	path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part1', 'part1c_D{}.csv' .format(depth))
# 	HF.write_out_tree(tree, path_to_outfile)


# # --- part 1d --- 