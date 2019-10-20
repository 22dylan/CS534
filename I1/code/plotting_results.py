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

	"""	
		generates a plot showing the SSE vs. the number of iterations
		used for report parts:
			-1a
			*1b
	"""


	path_to_pickle = os.path.join(os.getcwd(), '..', 'output', pickle_str)				# path to output
	data = pickle.load(open(path_to_pickle, "rb" ))

	num_keys = len(keys)
	fig, ax = plt.subplots(1, 1, figsize=(12,8))
	labels=[]

	linestyle = ['-', ':', '-.', '-']
	markers = ['.', '<', '^', 'v', '>', '+', 's']

	for i in range(num_keys):
		count = 0
		for key in keys[i]:
			data_temp = data[key]

			SSE = np.array([np.linalg.norm(data_temp['SSE'][i][1]) for i in range(len(data_temp['SSE']))]) #*68.9
			iteration = [data_temp['SSE'][i][0] for i in range(len(data_temp['SSE']))]

			if i == 0:
				labels.append(r'$\lambda$= {}' .format(data_temp['lambda_reg']))

			ax.plot(iteration, SSE, color='k', markevery=0.1, ls='-', marker=markers[count], linewidth=0.75)
			ax.grid(which='minor', alpha=0.25, color = 'k', ls = ':')
			ax.grid(which='major', alpha=0.40, color = 'k', ls = '--')
			count += 1

		ax.set_title('Training SSE')
		ax.set_ylabel('SSE')
		ax.set_xscale('log')
		ax.set_yscale('log')

	ax.set_xlabel('Iteration')
	plt.subplots_adjust(bottom=0.15)

	# black_patch = mpatches.Patch(color='k', label=r'Learning Rate: $10^{-5}$')
	# red_patch = mpatches.Patch(color='r', label=r'Learning Rate: $10^{-6}$')
	# blue_patch = mpatches.Patch(color='b', label=r'Learning Rate: $10^{-7}$')
	# ax.legend(handles=[black_patch, red_patch, blue_patch])


	fig.legend(labels = labels, loc="lower center", ncol=4)

def plot_w_values(pickle_str, keys):
	path_to_pickle = os.path.join(os.getcwd(), '..', 'output', pickle_str)				# path to output
	data = pickle.load(open(path_to_pickle, "rb" ))

	fig, ax = plt.subplots(1, 1, figsize=(12,8))
	w_str = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',\
	  		  'view','condition','grade','sqft_above','sqft_basement','yr_built',\
              'yr_renovated','zipcode', 'lat','long','sqft_living15','sqft_lot15',\
              'year', 'month', 'day']


	linestyle = ['-', ':', '-.', '-']
	markers = ['.', '<', '^', 'v', '>', '+', 's']
	for i, key in enumerate(keys):
		data_temp = data[key]

		w = data_temp['w_values']
		label_str = r'$\lambda: ${}' .format(data_temp['lambda_reg'])

		ax.plot(w, ls='-', marker=markers[i], color='k', label=label_str, linewidth=0.75)

	ax.grid(which='minor', alpha=0.25, color = 'k', ls = ':')
	ax.grid(which='major', alpha=0.40, color = 'k', ls = '--')
	ax.set_xticks(range(len(w_str)))
	ax.set_xticklabels(w_str, rotation=40, ha='right')
	
	# ax.set_yscale('log')
	fig.legend(loc="lower center", ncol=4)
	plt.subplots_adjust(bottom=0.2)

	ax.set_ylabel('w')


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
		print(len(data_temp['w_values']))
	print('\n')


def view_validation_results(pickle_str, keys):
	path_to_pickle = os.path.join(os.getcwd(), '..', 'output', pickle_str)
	data = pickle.load(open(path_to_pickle, "rb" ))
	for i in range(len(keys)):
		for key in keys[i]:
			data_temp = data[key]
			print(np.linalg.norm(data_temp))
	print('\n')

def view_training_SSE_results(pickle_str, keys):
	path_to_pickle = os.path.join(os.getcwd(), '..', 'output', pickle_str)
	data = pickle.load(open(path_to_pickle, "rb" ))
	SSE_ret = []
	for i in range(len(keys)):
		for key in keys[i]:
			SSE = data[key]['SSE'][-1][1]
			SSE_ret.append(np.linalg.norm(SSE))
			print(SSE_ret[-1])
	print('\n')

trials = [
			['trial_0', 'trial_1', 'trial_2', 'trial_3', 'trial_4', 'trial_5', 'trial_6']
		 ]

# trials = [
# 			['trial_35', 'trial_36', 'trial_37']
# 		 ]




# plot_SSE_v_Iterations(pickle_str='results_p2.pickle', keys=trials)
# plot_w_values(pickle_str = 'results_p2.pickle', keys=trials[0])

# view_W_values(pickle_str = 'OLD_results_p2.pickle', keys=['trial_6'])
view_validation_results(pickle_str='results_validation_p2.pickle', keys=[trials[0]])
view_training_SSE_results(pickle_str='results_p2.pickle', keys=[trials[0]])


plt.show()