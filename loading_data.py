import numpy as np
from numpy.random import shuffle
from math import log

all_data = np.loadtxt("WIFI.db/clean_dataset.txt")
shuffle(all_data)
training_number = int(0.8*len(all_data))
validation_end = int(0.9*len(all_data))
training, validation, testing = all_data[:training_number], all_data[training_number:validation_end], all_data[validation_end:]
print(training)
print(validation)
print(testing)






#for decision tree
source_1 = []
source_2 = []
source_3 = []
source_4 = []
source_5 = []
source_6 = []
source_7 = []
room_no = [] #LABEL

for datapoint in training:
    source_1.append(datapoint[0]) #all[datapoint][0]
    source_2.append(datapoint[1])
    source_3.append(datapoint[2])
    source_4.append(datapoint[3])
    source_5.append(datapoint[4])
    source_6.append(datapoint[5])
    source_7.append(datapoint[6])
    room_no.append(datapoint[7])

def label_entropy(labels):
    value,label_occurrences = np.unique(labels, return_counts=True)
    hits = label_occurrences / len(labels)
    return -(hits * np.log2(hits)).sum()

print("room number entropy: "+ str(label_entropy(room_no)))
