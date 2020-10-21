import numpy as np
from numpy.random import shuffle
from math import log

all_data = np.loadtxt("WIFI.db/clean_dataset.txt")
shuffle(all_data)
decile = 0.1*len(all_data)
training_number = int(8*decile)
validation_end = int(9*decile)
training, validation, testing = all_data[:training_number], all_data[training_number:validation_end], all_data[validation_end:]

print(training)
print(len(training))
print(validation)
print(testing)

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

#for decision tree
source_1 = []
source_2 = []
source_3 = []
source_4 = []
source_5 = []
source_6 = []
source_7 = []
room_no = [] #LABEL

for row in training:
    source_1.append(row[0]) #all[row][0]
    source_2.append(row[1])
    source_3.append(row[2])
    source_4.append(row[3])
    source_5.append(row[4])
    source_6.append(row[5])
    source_7.append(row[6])
    room_no.append(row[7])

value,label_occurrences = np.unique(room_no, return_counts=True)
hits = label_occurrences / len(training)
entropy_all = -(hits * np.log2(hits)).sum()
print("S(all): " + str(entropy_all))

left = []
right = []
for row in training:
    if(row[0] < -50):
        left.append(row)
    else:
        right.append(row)

value,label_occurrences = np.unique(room_no, return_counts=True)
hits = label_occurrences / len(left)
entropy_left = -(hits * np.log2(hits)).sum()
print("S(left): " + str(entropy_left))

value,label_occurrences = np.unique(room_no, return_counts=True)
hits = label_occurrences / len(right)
entropy_right = -(hits * np.log2(hits)).sum()
print("S(right): " + str(entropy_right))

remainder = (len(left)/(len(left)+len(right)))*entropy_left +(len(right)/(len(left)+len(right)))*entropy_right
information_gain = entropy_all - remainder

print("Information Gain: " + str(information_gain))
