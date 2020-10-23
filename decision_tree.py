import numpy as np
from numpy.random import shuffle
from math import log

all_data = np.loadtxt("WIFI.db/clean_dataset.txt")
shuffle(all_data)
decile = 0.1 * len(all_data)
training_number = int(8 * decile)
validation_end = int(9 * decile)
training = all_data[:training_number]
validation = all_data[training_number : validation_end]
testing =  all_data[validation_end:]

#Entropy of rooms
def entropy(array):
    rooms = []
    for row in array:
        rooms.append(row[7])
    room_number, label_occurrences = np.unique(rooms, return_counts = True)
    hits = label_occurrences / len(array)
    entropy = -(hits * np.log2(hits)).sum()
    return entropy

# #Entropy of whole dataset (Entropy(S))
# entropy_all = entropy(training)
# print("S(all): " + str(entropy_all))

#Finding split/threshold
def find_split(array):
    max_change = [-1,-1,-1] # [source_no, midpoint, remainder]
    rows, columns = array.shape[0], array.shape[1] - 1
    # Iterate over each column (not the last column of course, because it holds the room number)
    for i in range(columns):
        # Sort the entire array by the current column
        sorted_array = array[np.argsort(array[:,i])]
        # Find the points where Room changes value
        for j in range(rows - 1):
            if sorted_array[j, columns] != sorted_array[j + 1, columns]:
                # Take the midpoint of these two values in the current column
                midpoint = (sorted_array[j,i] + sorted_array[j + 1,i]) / 2
                # Find the Gain(midpoint, S)
                remainder = ((j + 1) / rows * entropy(sorted_array[:j + 1])) + ((rows - j + 1) / rows * entropy(sorted_array[j + 1:]))
                # If Gain > max_change.gain (this is the same as if remainder > max_change), max_change = midpoint, gain
                #print(i,midpoint, remainder)
                if remainder > max_change[2]:
                    max_change = [i, midpoint, remainder]
                    #print(" ----------------  max_change ----------------- ")
                    #print(max_change)
        # Continue until all elements have been read in that column and the max midpoint has been identified
    return max_change[0], max_change[1]

def decision_tree_learning(training, depth):
    # TODO: FIX THE IF CONDITION TO CHECK IF ALL SAMPLES HAVE SAME LABEL
    # TODO: LAST COLUMN OF TRAINING IS THE LABELS
    if True:
        node = {
            "attribute": None,
            "value": None,
            "left": None,
            "right": None,
            "leaf": True,
        }
        return node, depth
    else:
        attribute, split_value = find_split(training)
        
        left_set = []
        right_set = []
        for row in training:
            if(row[attribute - 1] < split_value):
                left_set.append(row)
            else:
                right_set.append(row)

        node = {
            "attribute": attribute,
            "value": split_value,
            "left": None,
            "right": None,
            "leaf": False,
        }

        node["left"], l_depth = decision_tree_learning(left_set, depth + 1)
        node["right"], r_depth = decision_tree_learning(right_set, depth + 1)
        return node, max(l_depth, r_depth)
