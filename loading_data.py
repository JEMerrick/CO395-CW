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

def entropy(array):
    rooms = []
    for row in array:
        rooms.append(row[7])
    value,label_occurrences = np.unique(rooms, return_counts=True)
    hits = label_occurrences / len(array)
    entropy = -(hits * np.log2(hits)).sum()
    return entropy

#for decision tree
entropy_all = entropy(training)
print("S(all): " + str(entropy_all))

left = []
right = []
for row in training:
    if(row[0] < -50): #I AM TESTING ATTRIBUTE SOURCE_1 < -50
        left.append(row)
    else:
        right.append(row)

entropy_left = entropy(left)
print("S(left): " + str(entropy_left))

entropy_right = entropy(right)
print("S(left): " + str(entropy_right))

remainder = (len(left)/(len(left)+len(right)))*entropy_left +(len(right)/(len(left)+len(right)))*entropy_right
information_gain = entropy_all - remainder

print("Information Gain: " + str(information_gain))
