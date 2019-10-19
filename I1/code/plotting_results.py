import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')

""" 
	code for viewing results.
	see excel sheet "output/pickle_summary.xlsx" for a summary
		of what's contained in each pickle file
"""



def plot_SSE_v_Iterations(pickle_str, keys):

	"""	
		generates a plot showing the SSE vs. the number of iterations
		used for report parts:
			-1a
			*1b
	"""


	path_to_pickle = os.path.join(os.getcwd(), '..', 'output', pickle_str)				# path to output

	data = pickle.load(open(path_to_pickle, "rb" ))

	num_subplots = len(keys)
	fig, ax = plt.subplots(num_subplots, 1, figsize=(12,8))
	labels=[]
	if num_subplots > 1:
		ax.flatten()
	for i in range(num_subplots):
		for key in keys[i]:
			data_temp = data[key]

			SSE = [np.linalg.norm(data_temp['SSE'][i][1]) for i in range(len(data_temp['SSE']))]
			iteration = [data_temp['SSE'][i][0] for i in range(len(data_temp['SSE']))]

			label_val = 'learning_rate: {}' .format(data_temp['lambda_reg'])
			if i == 0:
				labels.append('learning_rate: {}' .format(data_temp['lambda_reg']))
			ax[i].plot(iteration, SSE, label=label_val)
			ax[i].grid(which='minor', alpha=0.25, color = 'k', ls = ':')
			ax[i].grid(which='major', alpha=0.40, color = 'k', ls = '--')
		
		ax[i].set_title(r'$\lambda = {}$' .format(data_temp['step_size']))
		ax[i].set_ylabel('SSE')
		ax[i].set_xscale('log')

	ax[-1].set_xlabel('Iteration')
	plt.subplots_adjust(hspace=0.3, bottom=0.13)
	fig.legend(labels = labels, loc="lower center", ncol=4)

def view_W_values(pickle_str, keys):
	""" prints results contrained in each trial
		can be used for part:
			-1c
	"""

	path_to_pickle = os.path.join(os.getcwd(), '..', 'output', pickle_str)
	data = pickle.load(open(path_to_pickle, "rb" ))
	for key in keys:
		data_temp = data[key]
		print(key)
		print('\tstep_size: {}' .format(data_temp['step_size']))
		print('\tlambda_reg: {}' .format(data_temp['lambda_reg']))
		print('\tw_values: {}' .format(data_temp['w_values']))
	
	pass




trials = [['trial_0', 'trial_1', 'trial_2', 'trial_3', 'trial_4', 'trial_5', 'trial_6'],
			['trial_7', 'trial_8', 'trial_9', 'trial_10', 'trial_11', 'trial_12', 'trial_13'],
			['trial_14', 'trial_15', 'trial_16', 'trial_17', 'trial_18', 'trial_19', 'trial_20']
		 ]

plot_SSE_v_Iterations(pickle_str='results_p2_drs.pickle', keys=trials)
view_W_values(pickle_str = 'results_p2_drs.pickle', keys=['trial_4', 'trial_18'])


plt.show()