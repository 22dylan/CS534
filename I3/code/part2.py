import os
import numpy as np
import HelperFunctions as HF
import pandas as pd


# # -- all data ---
# path_to_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
# data = HF.datareader(path_to_data)
# features= list(data.columns[:-1])
# y = data.columns[-1]

# --- manageable test data ---
data = np.array([[0,0,0, 1],
				[0,0,1, 0],
				[0,1,0, 0],
				[0,1,1, 1],
				[1,0,0, 0],
				[1,0,1, 0],
				[1,1,0, 1],
				[1,1,1, 1]])

column_names = ['x1', 'x2', 'x3', 'f']
data = pd.DataFrame(data)
data.columns = column_names
y = column_names[-1]

df = data.drop(y, axis=1)

# --- Part 2a ---
""" 
    create a random forest with max depth d=2, num features in each tree m = 5, 
    and num trees in the forest n = 1, 2, 5, 10, 25  """
n = [1, 2, 5, 10, 25]
m = 2
depth = 2

for i in n:
    for ii in range(0, n[i]):
        sub_data = df.sample(len(data), replace=True, random_state=1) # sub-sample data
        sub_data = sub_data.sample(m, axis = 1) # choose features to keep
        sub_data = sub_data.join(data[y])
        features = sub_data.columns[0:-1]
        key = 'randfor_n='+str(ii)
        tree[key] = HF.make_tree(sub_data, depth, features, y)

        path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part2', 'TEST_M{}.csv' .format(depth))
        HF.write_out_tree(tree, path_to_outfile)

n = 15
m = [1 2 5 10 25 50]

for i in m:
    for ii in range(0, m[i]):
        sub_data = df.sample(len(data), replace=True, random_state=1) # sub-sample data
        sub_data = sub_data.sample(m(i), axis = 1) # choose features to keep
        sub_data = sub_data.join(data[y])
        features = sub_data.columns[0:-1]
        key = 'randfor_n='+str(ii)
        tree[key] = HF.make_tree(sub_data, depth, features, y)

        path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part2', 'TEST_M{}.csv' .format(depth))
        HF.write_out_tree(tree, path_to_outfile)