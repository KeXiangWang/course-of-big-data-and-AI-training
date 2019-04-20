import numpy as np


def load_file(filename):
    acc_file = open(filename)
    acc_data_file = []
    gyr_data_file = []
    acc_lines = acc_file.readlines()
    for line in acc_lines:
        line = line.replace('\n', '')
        line = line.replace('\r', '')
        numbers = line.split(' ')
        acc_data_line = np.array(numbers, dtype=float)
        acc_data_file.append(acc_data_line)
    for i in range(len(acc_data_file) - 1, 0, -1):
        acc_data_file[i][0] = acc_data_file[i][0] - acc_data_file[i - 1][0]
    acc_data_file[0][0] = 0
    return np.array(acc_data_file)


def load_data():
    acc_data = []
    gyr_data = []
    for i in range(10):
        acc_name = 'data/accData{}.txt'.format(i)
        gyr_name = 'data/gyrData{}.txt'.format(i)
        acc_data_file= load_file(acc_name)
        acc_data.append(acc_data_file)
        gyr_data_file= load_file(gyr_name)
        gyr_data.append(gyr_data_file)
    acc_data = np.array(acc_data)
    gyr_data = np.array(gyr_data)
    return acc_data, gyr_data
