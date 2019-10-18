import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')



# --- part 1 ---

def plot_1a(keys):
	"""
	trial_0 -> 10**0
	trial_1 -> 10**-1
	trial_2 -> 10**-2
	trial_3 -> 10**-3
	trial_4 -> 10**-4
	trial_5 -> 10**-5
	trial_6 -> 10**-6
	trial_7 -> 10**-7
	"""

	path_to_output = os.path.join(os.getcwd(), '..', 'output')				# path to output
	part_1_pickle = os.path.join(path_to_output, 'results_p1.pickle')

	fig, ax = plt.subplots(1, 1, figsize=(12,8))
	ax = np.array(ax)
	for key in keys:
		data = pickle.load(open(part_1_pickle, "rb" ))[key]

		label_val = 'learning_rate: {}' .format(data['step_size'])
		ax.plot(data['iteration'], data['SSE'], label='learning_rate: {}')
		ax.grid(which='minor', alpha=0.25, color = 'k', ls = ':')
		ax.grid(which='major', alpha=0.40, color = 'k', ls = '--')
		ax.set_xscale('log')

		print(data.keys())

	ax.set_ylabel('SSE')
	ax.set_xlabel('Iteration')
	# ax.set_xscale('log')


def plot_1b():
	pass

plot_1a(['trial_5', 'trial_6', 'trial_7'])		# trials 5, 6, and 7 don't diverge




plt.show()