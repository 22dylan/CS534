import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
""" 
	code for viewing results.
	see excel sheet "output/pickle_summary.xlsx" for a summary
		of what's contained in each pickle file
"""


def plot_SSE_v_Iterations(pickle_str, keys):
    path_to_pickle = os.path.join(os.getcwd(), '..', 'output', pickle_str)  # path to output
    data = pickle.load(open(path_to_pickle, "rb"))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    labels = []
    colors = ['k', 'r', 'b']

    for i in [ 5, 6, 7]:
        data_temp = data[keys[i]]

        SSE = np.array([np.linalg.norm(data_temp['SSE'][k][1]) for k in range(len(data_temp['SSE']))])  # *68.9
        iteration = [data_temp['SSE'][k][0] for k in range(len(data_temp['SSE']))]

        if i == 5:
            labels.append(r'$\lambda$= {}'.format(data_temp['lambda_reg']))

        ax.plot(iteration, SSE, color=colors[i-7], marker='.', markevery=0.1)

        ax.grid(which='minor', alpha=0.25, color='k', ls=':')
        ax.grid(which='major', alpha=0.40, color='k', ls='--')
        ax.set_title('Training SSE')
        ax.set_ylabel('SSE')
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    plt.subplots_adjust(bottom=0.15)

    black_patch = mpatches.Patch(color='k', label=r'Learning Rate: $10^{-5}$')
    red_patch = mpatches.Patch(color='r', label=r'Learning Rate: $10^{-6}$')
    blue_patch = mpatches.Patch(color='b', label=r'Learning Rate: $10^{-7}$')
    ax.legend(handles=[black_patch, red_patch, blue_patch])

    fig.legend(labels=labels, loc="lower center", ncol=4)


def view_W_values(pickle_str, keys):
    """ prints results contained in each trial
		can be used for part:
			-1c
	"""

    path_to_pickle = os.path.join(os.getcwd(), '..', 'output', pickle_str)
    data = pickle.load(open(path_to_pickle, "rb"))
    for key in keys:
        data_temp = data[key]
        print(key)
        print('\tstep_size: {}'.format(data_temp['step_size']))
        print('\tlambda_reg: {}'.format(data_temp['lambda_reg']))
        print('\tw_values: {}'.format(data_temp['w_values']))
    pass


def view_validation_results(pickle_str, keys):
    path_to_pickle = os.path.join(os.getcwd(), '..', 'output', pickle_str)
    data = pickle.load(open(path_to_pickle, "rb"))
    print(data.keys())
    for i in range(len(keys)):
        for key in keys[i]:
            data_temp = data[key]
            print(np.linalg.norm(data_temp))


def view_training_SSE_results(pickle_str, keys):
    path_to_pickle = os.path.join(os.getcwd(), '..', 'output', pickle_str)
    data = pickle.load(open(path_to_pickle, "rb"))
    print(data.keys())
    SSE_ret = []
    for i in range(len(keys)):
        SSE = data[i]['SSE'][-1][1]
        SSE_ret.append(np.linalg.norm(SSE))
        print(SSE_ret[-1])

    return SSE_ret


trials = ['trial_0', 'trial_1', 'trial_2', 'trial_3', 'trial_4', 'trial_5', 'trial_6', 'trial_7']

plot_SSE_v_Iterations(pickle_str='results_p1.pickle', keys=trials)
#view_W_values(pickle_str = 'results_p1.pickle', keys=trials)
#view_validation_results(pickle_str='results_validation_p1.pickle', keys=trials[0])
# view_training_SSE_results(pickle_str='results_p1.pickle', keys=trials[0])


plt.show()
