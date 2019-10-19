import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle

p = 0
# path = os.path.join(os.getcwd(), '..', 'data', 'PA1_train.csv')
path1 = os.path.join(os.getcwd(), '..', 'data', 'PA1_train.csv')
path2 = os.path.join(os.getcwd(), '..', 'data', 'PA1_test.csv')
path3 = os.path.join(os.getcwd(), '..', 'data', 'PA1_dev.csv')
path_to_output = os.path.join(os.getcwd(), '..', 'output', 'stats.pickle')  # path to output

train = pd.read_csv(path1)
train = train.drop('id', axis=1)  # remove id column

train['year'] = pd.DatetimeIndex(train['date']).year  # create day mo yr columns
train['month'] = pd.DatetimeIndex(train['date']).month
train['day'] = pd.DatetimeIndex(train['date']).day

test = pd.read_csv(path2)
test = test.drop('id', axis=1)  # remove id column
val = pd.read_csv(path3)
val = val.drop('id', axis=1)  # remove id column

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
            'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
            'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'price']
num_features = ['waterfront', 'condition', 'grade', 'view', 'day', 'month', 'year']
bw = [.6, .3, 0, 0, 0, 0, 0, 0, .3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# for statistics of features
train_stats = {};
val_stats = {};
test_stats = {};
pctwith = {}
for feat in features:
    train_stats[feat] = {}
    train_stats[feat]['mean'] = train[feat].mean()
    train_stats[feat]['std_dev'] = train[feat].std()
    train_stats[feat]['range'] = [train[feat].min(), train[feat].max()]
    print(feat, train_stats[feat])

    val_stats[feat] = {}
    val_stats[feat]['mean'] = val[feat].mean()
    val_stats[feat]['std_dev'] = val[feat].std()
    val_stats[feat]['range'] = [val[feat].min(), val[feat].max()]

    if feat != 'price':
        test_stats[feat] = {}
        test_stats[feat]['mean'] = test[feat].mean()
        test_stats[feat]['std_dev'] = test[feat].std()
        test_stats[feat]['range'] = [test[feat].min(), test[feat].max()]

    # plot pdfs
    if p == 1:
        plt.figure()
        if bw[features.index(feat)] == 0:
            train[feat].plot.kde()  # need to play w bandwidth on some of these
            val[feat].plot.kde()
            if feat != 'price':
                test[feat].plot.kde()
        else:
            train[feat].plot.kde(bw_method=bw[features.index(feat)])
            val[feat].plot.kde(bw_method=bw[features.index(feat)])
            test[feat].plot.kde(bw_method=bw[features.index(feat)])
        plt.ylabel(feat)
        plt.legend(['train', 'val', 'test'])
        plt.show()

for feat in num_features:
    pctwith[feat] = train[feat].value_counts(normalize=True)
    print(feat, pctwith[feat])

dicts = train_stats, val_stats, test_stats, pctwith
with open(path_to_output, 'wb') as file:
    pickle.dump(dicts, file, pickle.HIGHEST_PROTOCOL)
