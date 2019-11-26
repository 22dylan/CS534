import os
import numpy as np
import HelperFunctions as HF
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.stats as stats
import random

# -- all data ---
path_to_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
data = HF.datareader(path_to_data)
features = list(data.columns[:-1])
y = data.columns[-1]

# --- Part 2a ---
""" 
    create a random forest with max depth d=2, num features in each tree m = 5, 
    and num trees in the forest n = 1, 2, 5, 10, 25  """
n = [1, 2, 5, 10, 25]
m = 2
depth = 2
trees = []
df = pd.DataFrame()
# for i in n:
#     train_acc = []
#     val_acc = []
#     y_pred_trn_all = np.zeros((len(data), i))
#     y_pred_val_all = np.zeros((1625, i))
#     random.seed(8888)
#
#     for ii in range(i):
#         print(i, ii)
#         y_pred_trn = []
#         y_pred_val = []
#         shuff_data = data.sample(len(data), replace=True, random_state=1)  # sub-sample data
#         tree = HF.learn(shuff_data, features, y, depth, mode='DT', view_tree=False, m=m, bagging=1)
#         trees.append(tree)
#         path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part2', 'part2a', 'part2a_n{}_{}.csv'.format(i, ii))
#         HF.write_out_tree(tree, path_to_outfile)
#
#         # -- checking accuracy --
#         path_to_tree = os.path.join(os.getcwd(), '..', 'output', 'part2', 'part2a', 'part2a_n{}_{}.csv'.format(i, ii))
#         path_to_train_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
#         path_to_val_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_val.csv')
#         data_trn = HF.datareader(path_to_train_data)
#         data_val = HF.datareader(path_to_val_data)
#
#         _, y_pred_trn = HF.calc_error(path_to_tree, data_trn, y)  # get y prediction
#         _, y_pred_val = HF.calc_error(path_to_tree, data_val, y)
#
#         y_pred_trn_all[:, ii] = y_pred_trn[:]
#         y_pred_val_all[:, ii] = y_pred_val[:]
#
#     y_pred_trn_agg, _ = stats.mode(y_pred_trn_all, axis=1)
#     y_pred_val_agg, _ = stats.mode(y_pred_val_all, axis=1)
#
#     error_count_trn = HF.calc_error(path_to_tree, data_trn, y, mode='RF', y_pred=y_pred_trn_agg)  # send back to calc error
#     error_count_val = HF.calc_error(path_to_tree, data_val, y, mode='RF', y_pred=y_pred_val_agg)  # to get the error counts
#
#     acc_trn = 1 - (error_count_trn / len(data_trn))
#     acc_val = 1 - (error_count_val / len(data_val))
#
#     result = pd.DataFrame(
#         {'n': [i],
#          'train': [acc_trn],
#          'validation': [acc_val]
#          })
#     df = df.append(result)
#
# # write to csv
# path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part2', 'ValAcc_Results.csv')
# df.to_csv(path_to_outfile, index=False, na_rep="None")
#
# # plotting results
# fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# ax.plot(df['n'], df['train'], color='k', ls='-', label='Training')
# ax.plot(df['n'], df['validation'], color='k', ls='-.', label='Validation')
# ax.set_xlabel('number of trees in forest')
# ax.set_ylabel('Accuracy')
# ax.legend()
# ax.grid()
#
# plt.show()

#
# ____ part b _____
#
#
n = 15
m = [1, 2, 5, 10, 25, 50]

# for i in m:
#     train_acc = []
#     val_acc = []
#     y_pred_trn_all = np.zeros((len(data), n))
#     y_pred_val_all = np.zeros((1625, n))
#     random.seed(1337)
#
#     for ii in range(n):
#         print(i, ii)
#         y_pred_trn = []
#         y_pred_val = []
#         shuff_data = data.sample(len(data), replace=True)  # sub-sample data
#         tree = HF.learn(shuff_data, features, y, depth, mode='DT', view_tree=False, m=i, bagging=1)
#         trees.append(tree)
#         path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part2', 'part2b',
#                                        'part2b_m{}_n{}.csv'.format(i, ii))
#         HF.write_out_tree(tree, path_to_outfile)
#
#         # -- checking accuracy --
#         path_to_tree = os.path.join(os.getcwd(), '..', 'output', 'part2', 'part2b', 'part2b_m{}_n{}.csv'.format(i, ii))
#         path_to_train_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
#         path_to_val_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_val.csv')
#         data_trn = HF.datareader(path_to_train_data)
#         data_val = HF.datareader(path_to_val_data)
#
#         _, y_pred_trn = HF.calc_error(path_to_tree, data_trn, y)  # get y prediction
#         _, y_pred_val = HF.calc_error(path_to_tree, data_val, y)
#
#         y_pred_trn_all[:, ii] = y_pred_trn[:]
#         y_pred_val_all[:, ii] = y_pred_val[:]
#
#     y_pred_trn_agg, _ = stats.mode(y_pred_trn_all, axis=1)
#     y_pred_val_agg, _ = stats.mode(y_pred_val_all, axis=1)
#
#     error_count_trn = HF.calc_error(path_to_tree, data_trn, y, mode='RF',
#                                     y_pred=y_pred_trn_agg)  # send back to calc error
#     error_count_val = HF.calc_error(path_to_tree, data_val, y, mode='RF',
#                                     y_pred=y_pred_val_agg)  # to get the error counts
#
#     acc_trn = 1 - (error_count_trn / len(data_trn))
#     acc_val = 1 - (error_count_val / len(data_val))
#
#     result = pd.DataFrame(
#         {'m': [i],
#          'train': [acc_trn],
#          'validation': [acc_val]
#          })
#     df = df.append(result)
#
# # write to csv
# path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part2', 'ValAcc_Results2b.csv')
# df.to_csv(path_to_outfile, index=False, na_rep="None")
#
# # plotting results
# fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# ax.plot(df['m'], df['train'], color='k', ls='-', label='Training')
# ax.plot(df['m'], df['validation'], color='k', ls='-.', label='Validation')
# ax.set_xlabel('number of features trained on')
# ax.set_ylabel('Accuracy')
# ax.legend()
# ax.grid()
#
# plt.show()

#
# ____ part c _____
#
m=50
n=15
rseed = [1337, 1338, 1339, 88, 8888, 4444, 13, 7, 2, 9999]
for i in rseed:
    train_acc = []
    val_acc = []
    y_pred_trn_all = np.zeros((len(data), n))
    y_pred_val_all = np.zeros((1625, n))
    random.seed(i)

    for ii in range(n):
        print(i, ii)
        y_pred_trn = []
        y_pred_val = []
        shuff_data = data.sample(len(data), replace=True)  # sub-sample data
        tree = HF.learn(shuff_data, features, y, depth, mode='DT', view_tree=False, m=m, bagging=1)
        trees.append(tree)
        path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part2', 'part2c',
                                       'part2c_r{}_n{}.csv'.format(i, ii))
        HF.write_out_tree(tree, path_to_outfile)

        # -- checking accuracy --
        path_to_tree = os.path.join(os.getcwd(), '..', 'output', 'part2', 'part2c', 'part2c_r{}_n{}.csv'.format(i, ii))
        path_to_train_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_train.csv')
        path_to_val_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_val.csv')
        data_trn = HF.datareader(path_to_train_data)
        data_val = HF.datareader(path_to_val_data)

        _, y_pred_trn = HF.calc_error(path_to_tree, data_trn, y)  # get y prediction
        _, y_pred_val = HF.calc_error(path_to_tree, data_val, y)

        y_pred_trn_all[:, ii] = y_pred_trn[:]
        y_pred_val_all[:, ii] = y_pred_val[:]

    y_pred_trn_agg, _ = stats.mode(y_pred_trn_all, axis=1)
    y_pred_val_agg, _ = stats.mode(y_pred_val_all, axis=1)

    error_count_trn = HF.calc_error(path_to_tree, data_trn, y, mode='RF',
                                    y_pred=y_pred_trn_agg)  # send back to calc error
    error_count_val = HF.calc_error(path_to_tree, data_val, y, mode='RF',
                                    y_pred=y_pred_val_agg)  # to get the error counts

    acc_trn = 1 - (error_count_trn / len(data_trn))
    acc_val = 1 - (error_count_val / len(data_val))

    result = pd.DataFrame(
        {'rseed': [i],
         'train': [acc_trn],
         'validation': [acc_val]
         })
    df = df.append(result)

# write to csv
path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'part2', 'ValAcc_Results2c.csv')
df.to_csv(path_to_outfile, index=False, na_rep="None")

# plotting results
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(df['rseed'], df['train'], color='k', ls='-', label='Training')
ax.plot(df['rseed'], df['validation'], color='k', ls='-.', label='Validation')
ax.set_xlabel('number of features trained on')
ax.set_ylabel('Accuracy')
ax.legend()
ax.grid()

plt.show()