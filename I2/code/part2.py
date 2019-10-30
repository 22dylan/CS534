import os
import numpy as np


path_to_file = os.path.join('..', 'data', 'pa2_train.csv')
data = np.genfromtxt(path_to_file, delimiter =',')
data = data[0:1000, :]		# note: drs, remove this


TF_3 = data[:,0] == 3
TF_5 = data[:,0] == 5

data[TF_3, 0] = 1
data[TF_5, 0] = -1

data = np.column_stack((data, np.ones(len(data))))


iters = 15
w = np.zeros(np.shape(data)[1] - 1)
w_avg = np.zeros(np.shape(data)[1] - 1)

s = 1
itr = 0
while itr < iters:
	for i in range(len(data)):
		y_t = data[i,0]
		x_t = data[i,1:]
		if y_t*np.dot(w, x_t) <= 0:
			w += y_t*x_t

		w_avg = (s*w_avg + w)/(s+1)
		s += 1
	if s % 100 == 0:
		print(s)
	if s > 1000:
		break
print(w_avg)