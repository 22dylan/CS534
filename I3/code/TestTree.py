import os 
import pandas as pd
import HelperFunctions as HF



""" testing on training data """
y_pred_save = {}
depth_vals = [1, 2, 3, 4, 5, 6]
for depth in depth_vals:
	path_to_test_data = os.path.join(os.getcwd(), '..', 'data', 'pa3_test.csv')
	path_to_tree = os.path.join(os.getcwd(), '..', 'output', 'part1', 'part1c_D{}.csv' .format(depth))

	data_tst = HF.datareader(path_to_test_data)
	error_count_val, y_pred = HF.calc_error(path_to_tree, data_tst)


	y_pred_save['depth_{}' .format(depth)] = y_pred[:]

path_to_outfile = os.path.join(os.getcwd(), '..', 'output', 'pa3_test_predictions.csv')
df = pd.DataFrame(y_pred_save)
df.to_csv(path_to_outfile, index=False, na_rep="None")
