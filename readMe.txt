First of all, there is a 3 module in this project which are dataset.py, model.py and train.py.

In dataset.py, I am loading training, validation and testing data into loader as well as I am splitting train and validation set

In model.py, I have defined Models that are required in this assignment. There are 3 Models in this module explained in the description of the assignment.

In train.py, In the main function I'm creating new lists to determines the learning rates, layer counts, layer sizes as well as activation functions. After that I'have created a loop and called the run function after the getting the result from run function I'm writing these values into stats.txt files.
