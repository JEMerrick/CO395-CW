import numpy as np
from numpy.random import shuffle
from math import log

#Load data

all_data = np.loadtxt("WIFI.db/clean_dataset.txt")

#Splitting the training data up

shuffle(all_data)
decile = 0.1*len(all_data)
training_number = int(8*decile)
validation_end = int(9*decile)
training, validation, testing = all_data[:training_number], all_data[training_number:validation_end], all_data[validation_end:]


#Entropy of rooms
def entropy(array):
    rooms = []
    for row in array:
        rooms.append(row[7])
    room_number,label_occurrences = np.unique(rooms, return_counts=True)
    hits = label_occurrences / len(array)
    entropy = -(hits * np.log2(hits)).sum()
    return entropy

#FIRST SPLIT
entropy_all = entropy(training)
print("S(all): " + str(entropy_all))

def find_split(array):
    source_no = 1
    threshold = -50
    return source_no, threshold


left = []
right = []
for row in training:
    source_no, threshold = find_split(training)
    if(row[source_no-1] < threshold): #I AM TESTING ATTRIBUTE SOURCE_1 < -50
        left.append(row)
    else:
        right.append(row)

entropy_left = entropy(left)
print("S(left): " + str(entropy_left))

entropy_right = entropy(right)
print("S(right): " + str(entropy_right))

remainder = (len(left)/(len(left)+len(right)))*entropy_left +(len(right)/(len(left)+len(right)))*entropy_right
information_gain = entropy_all - remainder

print("Information Gain: " + str(information_gain))
