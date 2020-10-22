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

#Entropy of whole dataset (Entropy(S))
entropy_all = entropy(training)
print("S(all): " + str(entropy_all))

#Finding split/threshold
def find_split(array):
    maxChange = [-1,-1,-1] # [source_no, midpoint, remainder]
    rows, columns = array.shape[0], array.shape[1] - 1
    # Iterate over each column (not the last column of course, because it holds the room number)
    for i in range(columns):
        # Sort the entire array by the current column
        sortedArray = array[np.argsort(array[:,i])]
        # Find the points where Room changes value
        for j in range(rows - 1):
            if sortedArray[j,columns] != sortedArray[j+1,columns]:
                # Take the midpoint of these two values in the current column
                midpoint = (sortedArray[j,i] + sortedArray[j+1,i]) / 2
                # Find the Gain(midpoint, S)
                remainder = ((j + 1) / rows * entropy(sortedArray[:j+1])) + ((rows - j + 1) / rows * entropy(sortedArray[j+1:]))
                # If Gain > maxChange.gain (this is the same as if remainder > maxChange), maxChange = midpoint, gain
                print(i,midpoint, remainder)
                if remainder > maxChange[2]:
                    maxChange = [i, midpoint, remainder]
                    print(" ----------------  maxChange ----------------- ")
                    print(maxChange)
        # Continue until all elements have been read in that column and the max midpoint has been identified
    return maxChange[0], maxChange[1]


left = []
right = []
source_no, threshold = find_split(training)
for row in training:
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
