import numpy as np
from math import log, e

#how to split this into random 80/10/10 ratio?
#can't select last 10% as they will all be in room 4, not enough variation
training_set = np.loadtxt("WIFI.db/clean_dataset.txt")
print(all)

source_1 = []
source_2 = []
source_3 = []
source_4 = []
source_5 = []
source_6 = []
source_7 = []
room_no = [] #LABEL

for datapoint in training_set:
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
    hits = label_occurrences / len(training_set)
    return -(hits * np.log2(hits)).sum()

print("room number entropy: "+ str(label_entropy(room_no)))
