import numpy as np
from numpy.random import shuffle
from math import log

all_data = np.loadtxt("WIFI.db/clean_dataset.txt")
shuffle(all_data)
decile = 0.1*len(all_data)

#Validation
fold_1 = all_data[:int(decile)]
fold_2 = all_data[int(decile):int(2*decile)]
fold_3 = all_data[int(2*decile):int(3*decile)]
fold_4 = all_data[int(3*decile):int(4*decile)]
fold_5 = all_data[int(4*decile):int(5*decile)]
fold_6 = all_data[int(5*decile):int(6*decile)]
fold_7 = all_data[int(6*decile):int(7*decile)]
fold_8 = all_data[int(7*decile):int(8*decile)]
fold_9 = all_data[int(8*decile):int(9*decile)]
fold_10 = all_data[int(9*decile):]
