import numpy as np
from numpy.random import shuffle
import matplotlib.pyplot as plt
import matplotlib.lines as lines


def entropy(array):  # Entropy of rooms
    rooms = []
    for row in array:
        rooms.append(row[-1])  # Isolate last column (room no)
    room_number, label_occurrences = np.unique(rooms, return_counts=True)  #room no: 1 2 3 4
    hits = label_occurrences / len(array)  # label_occurrences = no. occurrences of room no: 1 2 3 4 respectvively
    entropy = -(hits * np.log2(hits)).sum()  # entropy = plog2(p)
    return entropy


def find_split(array):  # Finding split/threshold
    # print("find split")

    entropy_all = entropy(array)  # Entropy of whole dataset (Entropy(S))
    # print("S(all): " + str(entropy_all))

    max_change = [0, 0, 0, 0]  # [source_no, index, remainder, midpoint]
    rows, columns = array.shape[0], array.shape[1] - 1

    # Iterate over each column except last column as it holds the room number
    for i in range(columns):
        # Sort array by current column
        sorted_array = array[np.argsort(array[:, i])]

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
                remainder = (((j + 1) / rows) * entropy(sorted_array[:j + 1, :])) + (((rows - (j + 1)) / rows) * entropy(sorted_array[j + 1:, :]))
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
    initial = array[0][-1]  # Last element of first row (attribute)
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

    # Sort data
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

def accuracy(confusion_matrix, validation):
    # define room 1 as positive i.e. A[0][0]
    true_pos = confusion_matrix[0][0]
    true_neg = confusion_matrix[1][1] + confusion_matrix[2][2] + confusion_matrix[3][3]
    false_pos = confusion_matrix[1][0] + confusion_matrix[2][0] + confusion_matrix[3][0]
    false_neg = confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[0][3]

    # define room 2 as positive i.e. A[1][1]
    true_pos += confusion_matrix[1][1]
    true_neg += confusion_matrix[0][0] + confusion_matrix[2][2] + confusion_matrix[3][3]
    false_pos += confusion_matrix[0][1] + confusion_matrix[2][1] + confusion_matrix[3][1]
    false_neg += confusion_matrix[1][0] + confusion_matrix[1][2] + confusion_matrix[1][3]

    # define room 3 as positive i.e. A[2][2]
    true_pos += confusion_matrix[2][2]
    true_neg += confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[3][3]
    false_pos += confusion_matrix[0][2] + confusion_matrix[1][2] + confusion_matrix[3][2]
    false_neg += confusion_matrix[2][0] + confusion_matrix[2][1] + confusion_matrix[2][3]

    # define room 3 as positive i.e. A[3][3]
    true_pos += confusion_matrix[3][3]
    true_neg += confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2]
    false_pos += confusion_matrix[0][3] + confusion_matrix[1][3] + confusion_matrix[2][3]
    false_neg += confusion_matrix[3][0] + confusion_matrix[3][1] + confusion_matrix[3][2]

    true_pos /= 4
    true_neg /= 4
    false_pos /= 4
    false_neg /= 4

    accuracy = (true_pos + true_neg) / len(validation)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    F1 = (2 * precision * recall) / (precision + recall)

    return accuracy

def precision(confusion_matrix, validation):
    # define room 1 as positive i.e. A[0][0]
    true_pos = confusion_matrix[0][0]
    true_neg = confusion_matrix[1][1] + confusion_matrix[2][2] + confusion_matrix[3][3]
    false_pos = confusion_matrix[1][0] + confusion_matrix[2][0] + confusion_matrix[3][0]
    false_neg = confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[0][3]

    # define room 2 as positive i.e. A[1][1]
    true_pos += confusion_matrix[1][1]
    true_neg += confusion_matrix[0][0] + confusion_matrix[2][2] + confusion_matrix[3][3]
    false_pos += confusion_matrix[0][1] + confusion_matrix[2][1] + confusion_matrix[3][1]
    false_neg += confusion_matrix[1][0] + confusion_matrix[1][2] + confusion_matrix[1][3]

    # define room 3 as positive i.e. A[2][2]
    true_pos += confusion_matrix[2][2]
    true_neg += confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[3][3]
    false_pos += confusion_matrix[0][2] + confusion_matrix[1][2] + confusion_matrix[3][2]
    false_neg += confusion_matrix[2][0] + confusion_matrix[2][1] + confusion_matrix[2][3]

    # define room 3 as positive i.e. A[3][3]
    true_pos += confusion_matrix[3][3]
    true_neg += confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2]
    false_pos += confusion_matrix[0][3] + confusion_matrix[1][3] + confusion_matrix[2][3]
    false_neg += confusion_matrix[3][0] + confusion_matrix[3][1] + confusion_matrix[3][2]

    true_pos /= 4
    true_neg /= 4
    false_pos /= 4
    false_neg /= 4

    accuracy = (true_pos + true_neg) / len(validation)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    F1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, F1

def makeconfusion(node, validation):
    confusion_matrix = np.zeros(shape=(4, 4))
    for row in validation:
        actual_room = row[-1]
        # print("actual room: " + str(actual_room))
        predicted_room = 0
        predicted_room = traverse(node, predicted_room, row)
        # print("predicted room: " + str(predicted_room))
        confusion_matrix[int(actual_room-1)][int(predicted_room-1)] += 1

    return confusion_matrix

def evaluate(all_data, node):
    decile = 0.1 * len(all_data)

    # TODO K fold validation code

    # Minimum row no
    data_min = 0

    # Max row no
    data_max = all_data.shape[0]

    average_preprune_accuracy = 0
    average_prune_accuracy = 0

    average_preprune_precision = 0
    average_prune_precision = 0

    average_preprune_recall = 0
    average_prune_recall = 0

    average_preprune_F1 = 0
    average_prune_F1 = 0

    average_preprune_matrix = np.zeros(shape=(4, 4))
    average_prune_matrix = np.zeros(shape=(4, 4))

    for i in range(10):
        # increments
        x = int(i * decile)

        empty_matrix = np.zeros(shape=(4, 4))

        # Split into data ranges, A training (80%), B validation (10%), C testing (10%)
        A_start = x
        A_end = int(x + (8*decile))

        B_start = A_end
        B_end = int(B_start + decile)

        C_start = B_end
        C_end = int(C_start + decile)

        if(A_start > data_max):
            A_start = A_start - data_max

        if(B_start > data_max):
            B_start = B_start - data_max

        if(C_start > data_max):
            C_start = C_start - data_max

        if(A_end > data_max):
            A_end = A_end - data_max

        if(B_end > data_max):
            B_end = B_end - data_max

        if(C_end > data_max):
            C_end = C_end - data_max

        print("run number", i + 1)
        if(A_end > A_start):
            training = all_data[A_start:A_end]
            # print("A")
            # print(A_start, A_end)
        else:
            training = np.concatenate([all_data[A_start:data_max], all_data[data_min:A_end]])
            # print("A")
            # print(A_start, data_max, data_min, A_end)

        if(B_end > B_start):
            validation = all_data[B_start:B_end]
            # print("B")
            # print(B_start, B_end)
        else:
            validation = np.concatenate([all_data[B_start:data_max], all_data[data_min:B_end]])
            # print("B")
            # print(B_start, data_max, data_min, B_end)

        if(C_end > C_start):
            testing = all_data[C_start:C_end]
            # print("C")
            # print(C_start, C_end)
        else:
            testing = np.concatenate([all_data[C_start:data_max], all_data[data_min:C_end]])
            # print("C")
            # print(C_start, data_max, data_min, C_end)

        node, depth = decision_tree_learning(training, 0)
        confusion_matrix = makeconfusion(node, testing)
        average_preprune_matrix = np.add(confusion_matrix, average_preprune_matrix)
        preprune_accuracy = accuracy(confusion_matrix, testing)
        preprune_precision, preprune_recall, preprune_F1 = precision(confusion_matrix, testing)

        average_preprune_accuracy += preprune_accuracy
        average_preprune_precision += preprune_precision
        average_preprune_recall += preprune_recall
        average_preprune_F1 += preprune_F1

        #Finding the pruned accuracy
        prunedTree = prune(node, validation, training)
        pruned_confusion = makeconfusion(prunedTree, testing)
        average_prune_matrix = np.add(pruned_confusion, average_prune_matrix)
        pruned_accuracy = accuracy(pruned_confusion, testing)
        pruned_precision, pruned_recall, pruned_F1 = precision(pruned_confusion, testing)

        average_prune_accuracy += pruned_accuracy
        average_prune_precision += pruned_precision
        average_prune_recall += pruned_recall
        average_prune_F1 += pruned_F1

    average_preprune_accuracy /= 10
    average_prune_accuracy /= 10
    average_preprune_F1 /= 10
    average_prune_F1 /= 10
    average_prune_recall /= 10
    average_preprune_recall /= 10
    average_prune_precision /= 10
    average_preprune_precision /= 10
    average_prune_matrix /= 10
    average_preprune_matrix /= 10

    print("average preprune accuracy: " + str(average_preprune_accuracy))
    print("average prune accuracy: " + str(average_prune_accuracy))
    print("average preprune recall: " + str(average_preprune_recall))
    print("average prune recall: " + str(average_prune_recall))
    print("average preprune precision: " + str(average_preprune_precision))
    print("average prune precision: " + str(average_prune_precision))
    print("average preprune F1: " + str(average_preprune_F1))
    print("average prune F1: " + str(average_prune_F1))
    print("average preprune confusion matrix: ")
    print(average_preprune_matrix)
    print("average prune confusion matrix: ")
    print(average_prune_matrix)
    return average_preprune_accuracy, average_prune_accuracy

def traverse(node, room, test_row):
    # print(test_row)
    # print("node")
    # print(node["value"])
    # print(node["attribute"])

    # if node value == none, we are at a leaf node
    if(node["value"] is None):
        room = node["attribute"]
        # print("room is ")
        # print(room)
    else:
        # node[attribute] = column being tested, test_row[node[attribute]] = value in test row in that column that we want to compare
        if(test_row[node["attribute"]] < node["value"]):
            room = traverse(node["left"], room, test_row)
        else:
            room = traverse(node["right"], room, test_row)

    return room

def prune(node, validation, training):

    #TODO I dont actually need this function anymore, it can be removed for cleaner code later

    #Find the accuracy of the current tree
    confusion_matrix = makeconfusion(node, validation)
    Accuracy = accuracy(confusion_matrix, validation)

    #Find the new tree and accuracy
    newTree = depth_search(node, training, validation, node)

    confusion_matrix = makeconfusion(newTree, validation)
    newAccuracy = accuracy(confusion_matrix, validation)


    #print("############### NEW TREE #############")
    #print(newTree)
    #print(" ######   original accuracy    #####")
    #print(Accuracy)
    #print("#####    new accuracy    ######")
    #print(newAccuracy)


    return newTree


def depth_search(node, training, validation, rootNode):

    #we are searching for a parent node with two children that are leaves
    #If we reach a leaf move back up the tree, this has the effect that if a node is pruned, it will be checked again by the function
    if(node["leaf"] == True):
        return node
    else:
        if((node["left"]["leaf"] == True) and (node["right"]["leaf"] == True)):

            #Finding accuracy of original node
            confusion_matrix = makeconfusion(rootNode, validation)
            Accuracy = accuracy(confusion_matrix, validation)

            #print("old accuracy")
            #print(Accuracy)

            #Save the old node incase
            oldNode = node.copy()

            #Remove the leaf nodes, set the current node as a leaf
            node["left"] = None
            node["right"] = None
            node["leaf"] = True
            node["value"] = None

            #Find which room has the most occurances set = attribute
            #Need the set of data associated with each node
            #Taking column of rooms only
            rooms = training[:,-1].astype(int)
            frequentRoom = np.argmax(np.bincount(rooms.flat))
            node["attribute"] = frequentRoom

            #Test the accuracy here
            confusion_matrix = makeconfusion(rootNode, validation)
            newAccuracy = accuracy(confusion_matrix, validation)

            #print("new accuracy")
            #print(newAccuracy)

            #If accuracy decreased undo prune
            if(newAccuracy <= Accuracy):
                node = oldNode.copy()

            #Else do nothing

        else:
            #Keep traversing until reached leaves

            #splitting the data as we move through nodes
            attribute, index, split_value = find_split(training)

            sorted_data = training[np.argsort(training[:, attribute])]
            left_set = sorted_data[:index + 1, :]
            right_set = sorted_data[index + 1:, :]

            #search the left and right branches
            node["left"] = depth_search(node["left"], left_set, validation, rootNode)
            node["right"] = depth_search(node["right"], right_set, validation, rootNode)

    return node

def visualise_tree(node):
    fig = plt.figure()
    # x,y are parent coords, x1,y1 are current node coords
    x = 0
    y = 0
    x1 = 0.5
    y1 = 0.9

    print_tree(node, fig, x, y, x1, y1)

    plt.show()


def print_tree(root, fig, x, y, x1, y1):

    if root is None:
        return

    print_tree(root['right'], fig, x, y, x1 - 0.08, y1 - 0.05)

    # Process left child
    print_tree(root['left'], fig, x, y, x1 + 0.08, y1 - 0.05)

    fig.text(x1, y1, root['attribute'], ha='center', va='center')
    # plt.plot([x,x1], [y,y1], 'k-')


def main():

    all_data = np.loadtxt("WIFI.db/noisy_dataset.txt")

    shuffle(all_data)

    decile = 0.1 * len(all_data)
    training_number = int(8 * decile)
    validation_end = int(9 * decile)

    training = all_data[:training_number]
    validation = all_data[training_number:validation_end]
    testing = all_data[validation_end:]

    node, depth = decision_tree_learning(training, 0)

    evaluate(all_data, node)

    #print("-----PRINT TREE------\n\n\n")

    #visualise_tree(node)


main()
