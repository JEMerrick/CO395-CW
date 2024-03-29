# CO395 - Introduction to Machine Learning - Coursework 1

Authors:
- Joanna Merrick
- George Chen
- Jason Keung
- Jason Zheng

There are 4 functions in our code:

1. decision_tree_learning()
  To call this function, you must input a dataset and depth (set depth = 0)
  This function will return the root node (so the whole tree), and the max depth.
  
2. evaluate()
  Evaluate and prune are interlinked, as the nested 10 fold cross validation calls prune.
  Evaluate will perform the nested 10 fold cross validation. It takes approximately 30 seconds to create, prune and find the metrics per tree. 
  The function will create/test 90 trees. Therefore when running this function please note it will take approximately 45 minutes to complete.
  
  To call this function, you must input a dataset and tree (root node)
  This function will return average preprune accuracy and average pruned accuracy.
  This function will also print the confusion matrix, accuracy, recall, precision, f1, and depth for before and after pruning to the console.

3. depth_search()
  
  This is our prune function. If you wish to prune a tree on its own you can call this function, but you need not call it if you want the 10 fold cross validation.
  
  To call this function, you must input the tree (root node), a validation set, the root node (this should be the same as the first value), and depth (set depth = 0) 
  depth_search will output the pruned tree (root node) and its depth.
  
4. create_plot()
  
  This is our visualise tree function. If you wish to draw the tree you can call this.
  
  To call this function, you must input your tree and depth. 
  The most logical way to do this is to first create a tree by calling decision_tree_learning, which will output a tree and depth. After that you can use those values to call the create plot function.
  
  This function will output a .png file of the tree given called "tree.png".

Calling the main function will shuffle the data, then call decision_tree_learning(), create_plot(), and evaluate().

To edit the dataset, please edit the filepath in the call to np.loadtxt() in the main.