import pickle
from BatchGradientDescent import bgd

def run_bgd(data, step_size_vals, lambda_vals, path_to_pickle):
	w = {}			# preallocating a dictionary
	count = 0
	for step_size in step_size_vals:			# loop through step sizes
		for lambda_reg in lambda_vals:			# loop through regularization values
			trial_str = 'trial_{}' .format(count)	# sring for dictionary
			w[trial_str] = {}						# setting up dictionary key
			w[trial_str]['step_size'] = step_size 	# storing step size in dictionary
			w[trial_str]['lambda_reg'] = lambda_reg	# storing lambda (regularizing) value in dictionary
			
			print('\tstep_size: {}, lambda: {}' .format(step_size, lambda_reg))

			w[trial_str]['w_values'], w[trial_str]['convergence_count'], w[trial_str]['SSE'] =  bgd(lambda_reg, step_size,data)
			count += 1
	# saving results to a pickle
	print(w.keys())
	with open(path_to_pickle, 'wb') as f:
		pickle.dump(w, f, pickle.HIGHEST_PROTOCOL)

	return w