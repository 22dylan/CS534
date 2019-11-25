import os
import numpy as np
import HelperFunctions as HF
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
np.random.seed(47657)
# -- all data ---
path_to_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
data = HF.datareader(path_to_data)
features = list(data.columns[:-1])
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
#
# column_names = ['x1', 'x2', 'x3', 'f']
# data = pd.DataFrame(data)
# data.columns = column_names
# y = column_names[-1]



# --- Part 2a ---
""" 
    create a random forest with max depth d=2, num features in each tree m = 5, 
    and num trees in the forest n = 1, 2, 5, 10, 25  """
n = [1, 2, 5, 10, 25]
m = 2
depth = 2
trees = []
best_tree = {}
train_acc = []
val_acc = []
for i in n:
    for ii in range(i):
        print (i, ii)
        shuff_data = data.sample(len(data), replace=True, random_state=1)  # sub-sample data
        tree = HF.learn(shuff_data, features, y, depth, mode='DT', view_tree=False, m=m, bagging=1)
        trees.append(tree)
        path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part2', 'part2a_n{}_{}.csv'.format(i, ii))
        HF.write_out_tree(tree, path_to_outfile)

        # -- checking accuracy --
        path_to_tree = os.path.join(os.getcwd(), '..', 'output', 'part2', 'part2a_n{}_{}.csv'.format(i, ii))
        path_to_train_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
        path_to_val_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_val.csv')
        data_trn = HF.datareader(path_to_train_data)
        data_val = HF.datareader(path_to_val_data)

        error_count_trn, _ = HF.calc_error(path_to_tree, data_trn, y)
        error_count_val, _ = HF.calc_error(path_to_tree, data_val, y)
        acc_trn = 1 - (error_count_trn / len(data_trn))
        acc_val = 1 - (error_count_val / len(data_val))

        train_acc.append(acc_trn)
        val_acc.append(acc_val)
    ind=np.argmax(val_acc)
    best_tree['n={}'.format(i)] = trees[ind]




# writing to csv
path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part2', 'ValAcc_Results.csv')
df = pd.DataFrame(
    {'n': n,
     'train': train_acc,
     'validation': val_acc
    })
df.to_csv(path_to_outfile, index=False, na_rep="None")

# plotting results
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(n, train_acc, color='k', ls='-', label='Training')
ax.plot(n, val_acc, color='k', ls='-.', label='Validation')
ax.set_xlabel('number of trees in forest')
ax.set_ylabel('Accuracy')
ax.legend()
ax.grid()


plt.show()

n = 15
m = [1, 2, 5, 10, 25, 50]

for i in m:
    for ii in range(0, m[i]):
        shuff_data = data.sample(len(data), replace=True, random_state=1)  # sub-sample data
        tree = HF.learn(shuff_data, features, y, depth, mode='DT', view_tree=False, m=m, bagging=1)
        trees.append(tree)
        path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part2', 'part2b_m{}.csv'.format(m[i]))
        HF.write_out_tree(trees, path_to_outfile)


# -- checking accuracy --



for num in m:
	path_to_tree = os.path.join(os.getcwd(), '..', 'output', 'part2', 'part2b_m{}.csv' .format(m[num]))
	y = data_trn.columns[-1]
	error_count_trn, _ = HF.calc_error(path_to_tree, data_trn, y)
	error_count_val, _ = HF.calc_error(path_to_tree, data_val, y)

	acc_trn = 1 - (error_count_trn/len(data_trn))
	acc_val = 1 - (error_count_val/len(data_val))

	train_acc.append(acc_trn)
	val_acc.append(acc_val)

# writing to csv
path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part2', 'ValAcc_Results.csv')
df = pd.DataFrame(
    {'m': m,
     'train': train_acc,
     'validation': val_acc
    })
df.to_csv(path_to_outfile, index=False, na_rep="None")

# plotting resultsr
fig, ax = plt.subplots(1,1, figsize=(10, 6))
ax.plot(m, train_acc, color='k', ls='-', label='Training')
ax.plot(m, val_acc, color='k', ls='-.', label='Validation')
ax.set_xlabel('number of features in trees')
ax.set_ylabel('Accuracy')
ax.legend()
ax.grid()


plt.show()
