import data_process
import numpy as np
import os

if __name__ == "__main__":
    if os.path.exists("acc_data.npy"):
        acc_data = np.load("acc_data.npy")
        gyr_data = np.load("gyr_data.npy")
        print("Acc_data and Gyr_data loaded: ", acc_data.shape, gyr_data.shape)
    else:
        acc_data, gyr_data = data_process.load_data()
        np.save("acc_data.npy", acc_data)
        np.save("gyr_data.npy", gyr_data)
    print("Hello World!")
