import data_process
from tree import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    acc_sequence, gyr_sequence, label_sequence = data_process.get_sequence()
    # tree(acc_sequence, gyr_sequence, label_sequence)
    print("Hello World!")
