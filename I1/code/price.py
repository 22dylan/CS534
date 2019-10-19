# inputs: trial name, data set, and pickle file of results
# returns: price
import pickle

def price(trials, test_data, picklepath):
    results = pickle.load(open(picklepath, 'rb'))
    price = {}
    for trial_num in trials:
        w = results[trial_num]['w_values']
        price[trial_num] = test_data.dot(w[trial_num])

    return price