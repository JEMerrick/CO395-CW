import numpy as np
from numpy.random import shuffle
from math import log

all_data = np.loadtxt("WIFI.db/clean_dataset.txt")

shuffle(all_data)
decile = 0.1 * len(all_data)
training_number = int(8 * decile)
validation_end = int(9 * decile)
training, validation, testing = all_data[:training_number], all_data[training_number:validation_end], all_data[validation_end:]


#Entropy of rooms
def entropy(array):
    rooms = []
    for row in array:
        rooms.append(row[-1]) #isolate last column (room no)
    room_number, label_occurrences = np.unique(rooms, return_counts = True) #room no: 1 2 3 4
    hits = label_occurrences / len(array) #label_occurrences = no. occurrences of room no: 1 2 3 4 respectvively
    entropy = -(hits * np.log2(hits)).sum() #entropy = plog2(p)
    return entropy

# #Entropy of whole dataset (Entropy(S))
# entropy_all = entropy(training)
# print("S(all): " + str(entropy_all))

#Finding split/threshold
def find_split(array):
    print("find split")

    #Entropy of whole dataset (Entropy(S))
    entropy_all = entropy(array)
    print("S(all): " + str(entropy_all))

    maxChange = [0,0,0] # [source_no, midpoint, remainder]
    rows, columns = array.shape[0], array.shape[1] - 1
    # Iterate over each column (not the last column of course, because it holds the room number)
    if(rows == 2):
        for i in range(columns):
            sortedArray = array[np.argsort(array[:,i])]
            for j in range(rows - 1):
                midpoint = (sortedArray[j,i] + sortedArray[j+1,i]) / 2
                diff = abs(sortedArray[j,i] - sortedArray[j+1,i])
                if(maxChange[2] < diff):
                    maxChange = [i, midpoint, diff]
    else:
        for i in range(columns):
            # Sort the entire array by the current column
            sortedArray = array[np.argsort(array[:,i])]
            # Find the points where Room changes value
            for j in range(rows - 1):
                if sortedArray[j,columns] != sortedArray[j+1,columns]:
                    # Take the midpoint of these two values in the current column
                    midpoint = (sortedArray[j,i] + sortedArray[j+1,i]) / 2
                    # Find the Gain(midpoint, S)
                    remainder = (((j + 1) / rows) * entropy(sortedArray[:j+1,:])) + (((rows - (j + 1)) / rows) * entropy(sortedArray[j+1:,:]))
                    # If Gain > maxChange.gain (this is the same as if remainder > maxChange), maxChange = midpoint, gain

                    if ((entropy_all - remainder) > maxChange[2]):
                        print(j)
                        maxChange = [i, midpoint, remainder]
                        print(" ----------------  maxChange ----------------- ")
                        print(maxChange)
            # Continue until all elements have been read in that column and the max midpoint has been identified
    return maxChange[0], maxChange[1]


def label_same(array):
    print("array")
    print(array)
    initial = array[0][-1] # Last element of first row (attribute)
    for row in array:
        if row[-1] != initial:
            return False
    return True

def decision_tree_learning(training, depth):
    print("decision tree")
    if label_same(training):
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
        print("attribute")
        print(attribute)
        print("split_value")
        print(split_value)
        left_set = []
        right_set = []
        for row in training:
            if(row[attribute] <= split_value):
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

        left_set = np.array(left_set)
        right_set = np.array(right_set)

        node["left"], l_depth = decision_tree_learning(left_set, depth + 1)
        node["right"], r_depth = decision_tree_learning(right_set, depth + 1)
        return node, max(l_depth, r_depth)




node, depth = decision_tree_learning(training, 0)
print(node)
print("Depth is ", depth)
