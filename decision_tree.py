import numpy as np
from numpy.random import shuffle
from math import log

# class Node:
#     def __init__(self, attribute, value, left, right, leaf):
#         self.attribute  = 
#         self.value = 
#         self.left = None
#         self.right = None
#         self.leaf = False

all_data = np.loadtxt("WIFI.db/clean_dataset.txt")
shuffle(all_data)
decile = 0.1*len(all_data)
training_number = int(8*decile)
validation_end = int(9*decile)
training, validation, testing = all_data[:training_number], all_data[training_number:validation_end], all_data[validation_end:]

def find_split(training): #this function chooses the attribute and the value that results in the highest information gain
    # Sort the values of the attribute and then consider only split points taht are between two examples in sorted order
    # while keeping track of running totals of positive and negative examples on each side of split point

    # Check information gain + entropy here and select the biggest one
    return

def decision_tree_learning(training, depth): # use dicts to store nodes: node has attribute, value, left, right (left and right are both nodes), can add bool to say if leaf or not
    node = {
        "attribute" = None,
        "value" = None,
        "left" = None,
        "right" = None,
        "leaf" = False,
    }

    # if all samples have same label: CHECK LAST COLUMN OF TRAINING
        return node
    else:
        split = find_split(training)
        # node = new node with root = split
        l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
        r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
        return node, max(l_depth, r_depth)

