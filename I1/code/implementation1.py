import os
import numpy as np
from run_bgd import run_bgd
from data_reader import data_reader


""" loading in data both normalized and raw """

path = os.path.join(os.getcwd(), '..', 'data', 'PA1_train.csv')
normed_data = data_reader(path,norm=True)
raw_data = data_reader(path,norm=False)

# --- part 1 --- 
""" description here """
step_size = [10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7]
lambda_vals = [0]
w_p1 = run_bgd(normed_data, step_size, lambda_vals)

# --- part 2 ---
""" description here """
step_size = [10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7]
lambda_vals = [0, 10**-3, 10**-2, 10**-1, 1, 10, 100]
w_p2 = run_bgd(normed_data, step_size, lambda_vals)

# --- part 3 ---
step_size = [10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7]
lambda_vals = [0, 10**-3, 10**-2, 10**-1, 1, 10, 100]
w_p3 = run_bgd(raw_data, step_size, lambda_vals)



# --- testing/validating models ---
"""  
	to-do: 
	-write a function that uses the calculated
		w's on the test/validation data. 
	-make some nice plots
"""


