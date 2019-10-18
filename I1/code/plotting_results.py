import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')



path_to_output = os.path.join(os.getcwd(), '..', 'output')				# path to output
part_1_pickle = os.path.join(path_to_output, 'results_p1.pickle')
part_2_pickle = os.path.join(path_to_output, 'results_p2.pickle')
part_3_pickle = os.path.join(path_to_output, 'results_p3.pickle')


# --- part 1 ---
results = pickle.load(open(part_1_pickle, "rb" ))


learning_rates = [results[i]['step_size'] for i in results.keys()]
SSE = [results[i]['SSE'] for i in results.keys()]
convergence_count = [results[i]['convergence_count'] for i in results.keys()]
print(results['trial_2'])

fig, ax = plt.subplots(2, 1, figsize=(12,8))
ax = np.array(ax)

ax[0].plot(learning_rates, SSE, color='k', lw=2, ls = '-')
ax[1].plot(learning_rates, convergence_count, color='k', lw=2, ls = '-')


ax[0].grid(which='minor', alpha=0.25, color = 'k', ls = ':')
ax[0].grid(which='major', alpha=0.40, color = 'k', ls = '--')
ax[1].grid(which='minor', alpha=0.25, color = 'k', ls = ':')
ax[1].grid(which='major', alpha=0.40, color = 'k', ls = '--')

ax[0].set_ylabel('SSE')
ax[1].set_ylabel('Convergence Count')
ax[1].set_xlabel('Learning Rate')

ax[0].set_xscale('log')
ax[1].set_xscale('log')





# # --- part 2 ---
# results = pickle.load(open(part_2_pickle, "rb" ))
# learning_rates = [results[i]['step_size'] for i in results.keys()]
# SSE = [results[i]['SSE'] for i in results.keys()]
# convergence_count = [results[i]['convergence_count'] for i in results.keys()]

# fig, ax = plt.subplots(2, 1, figsize=(12,8))
# ax = np.array(ax)

# ax[0].plot(learning_rates, SSE, color='k', lw=2, ls = '-')
# ax[1].plot(learning_rates, convergence_count, color='k', lw=2, ls = '-')


# ax[0].grid(which='minor', alpha=0.25, color = 'k', ls = ':')
# ax[0].grid(which='major', alpha=0.40, color = 'k', ls = '--')
# ax[1].grid(which='minor', alpha=0.25, color = 'k', ls = ':')
# ax[1].grid(which='major', alpha=0.40, color = 'k', ls = '--')

# ax[0].set_ylabel('SSE')
# ax[1].set_ylabel('Convergence Count')
# ax[1].set_xlabel('Learning Rate')

# ax[0].set_xscale('log')
# ax[1].set_xscale('log')
# plt.show()




plt.show()