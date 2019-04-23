import data_process
from tree import *
from SVM import *
import numpy as np

if __name__ == "__main__":
    acc_sequence, gyr_sequence, label_sequence = data_process.get_sequence()
    # tree(acc_sequence, gyr_sequence, label_sequence)
    # SVM(acc_sequence, gyr_sequence, label_sequence)
    print("Hello World!")
