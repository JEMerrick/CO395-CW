import numpy as np
from numpy.random import shuffle
from math import log

def entropy(array): # Entropy of rooms
    rooms = []
    for row in array:
        rooms.append(row[-1]) # Isolate last column (room no)
    room_number, label_occurrences = np.unique(rooms, return_counts = True) #room no: 1 2 3 4
    hits = label_occurrences / len(array) # label_occurrences = no. occurrences of room no: 1 2 3 4 respectvively
    entropy = -(hits * np.log2(hits)).sum() # entropy = plog2(p)
    return entropy

def find_split(array): # Finding split/threshold
    # print("find split")

    entropy_all = entropy(array) # Entropy of whole dataset (Entropy(S))
    # print("S(all): " + str(entropy_all))

    max_change = [0, 0, 0, 0] # [source_no, index, remainder, midpoint]
    rows, columns = array.shape[0], array.shape[1] - 1
    # Iterate over each column (not the last column of course, because it holds the room number)

    for i in range(columns):
        sorted_array = array[np.argsort(array[:, i])] # Sort the entire array by the current column
        # Find the points where Room changes value
        for j in range(rows - 1):
            if rows == 2:
                midpoint = (sorted_array[j, i] + sorted_array[j + 1, i]) / 2
                diff = abs(sorted_array[j, i] - sorted_array[j + 1, i])
                if(max_change[2] < diff):
                    max_change = [i, j, diff, midpoint]

            elif sorted_array[j, columns] != sorted_array[j + 1, columns]:
                # Take the midpoint of these two values in the current column
                midpoint = (sorted_array[j, i] + sorted_array[j + 1, i]) / 2

                # Find the Gain(midpoint, S)
                remainder = (((j + 1) / rows) * entropy(sorted_array[:j + 1, :])) + (((rows - (j + 1)) / rows) * entropy(sorted_array[j + 1:, :])) + 0
                gain = entropy_all - remainder

                # If Gain > max_change.gain max_change = midpoint, gain
                if(gain > max_change[2]):
                    max_change = [i, j, gain, midpoint]
                    # print(" ----------------  max_change ----------------- ")
                    # print(max_change)
            # Continue until all elements have been read in that column and the max midpoint has been identified
    return max_change[0], max_change[1], max_change[3]

def label_same(array):
    # print("array")
    # print(array)
    initial = array[0][-1] # Last element of first row (attribute)
    for row in array:
        if row[-1] != initial:
            return False
    return True

def decision_tree_learning(training, depth):
    # print("decision tree")
    if label_same(training):
        node = {
            "attribute": training[0][-1],
            "value": None,
            "left": None,
            "right": None,
            "leaf": True,
        }
        return node, depth

    attribute, index, split_value = find_split(training)

    # print("attribute")
    # print(attribute)

    # print("split_value")
    # print(split_value)

    # print(training)

    #sort data
    sorted_data = training[np.argsort(training[:, attribute])]
    left_set = sorted_data[:index + 1, :]
    right_set = sorted_data[index + 1:, :]
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

def evaluate():
    all_data = np.loadtxt("WIFI.db/clean_dataset.txt")

    shuffle(all_data)
    
    decile = 0.1 * len(all_data)
    training_number = int(8 * decile)
    validation_end = int(9 * decile)
    
    for i in range(10):
        training = all_data[:training_number + decile]
        validation = all_data[training_number + decile:validation_end + decile]
        testing = all_data[validation_end + decile:]
        
        

def main():
    

    

    

    node, depth = decision_tree_learning(training, 0)
    print(node)
    print("Depth is ", depth)

    test_row = validation[0]
    actual_room = validation[0][7]
    while(node["leaf"] == True):
        if(test_row[node["attribute"]] < node["value"]):
            node = node["left"]
        else:
            node = node["right"]
    predicted_room = node["attribute"]

main()
