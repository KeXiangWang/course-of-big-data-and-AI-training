import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

WINDOW_LENGTH = 200
WINDOW_GAP = 100

USE_OLD_DATA = True
USE_OLD_SEQ = True


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
        acc_data_file[i][0] = acc_data_file[i][0] - acc_data_file[0][0]
    acc_data_file[0][0] = 0
    return np.array(acc_data_file)


def load_data():
    acc_data = []
    gyr_data = []
    for i in range(10):
        acc_name = 'data/accData{}.txt'.format(i)
        gyr_name = 'data/gyrData{}.txt'.format(i)
        acc_data_file = load_file(acc_name)
        acc_data.append(acc_data_file)
        gyr_data_file = load_file(gyr_name)
        gyr_data.append(gyr_data_file)
    acc_data = np.array(acc_data)
    gyr_data = np.array(gyr_data)
    return acc_data, gyr_data


def data_filter(acc_data):
    b, a = signal.butter(4, 0.1, 'lowpass')
    x, y, z = acc_data.shape
    data_out = []
    for i in range(x):
        data_line = []
        for j in range(0, z):
            filtedData = signal.filtfilt(b, a, acc_data[i, :, j])  # data为要过滤的信号
            data_line.append(filtedData)
        data_line = np.transpose(np.array(data_line))
        # print(data_out.shape)
        data_out.append(data_line)
    data_out  = np.array(data_out)
    # print(data_out.shape)
    return data_out[:, 500:, 1:]


def plot_data(acc_data, gyr_data):
    axis_0 = 8
    axis_1_start = 500
    axis_1_end = 1000
    axis_2 = 1
    plt.plot(acc_data[axis_0, axis_1_start:axis_1_end, axis_2])
    # b, a = signal.butter(8, 0.2, 'lowpass')
    # filtedData = signal.filtfilt(b, a, acc_data[axis_0, axis_1_start:axis_1_end, axis_2])  # data为要过滤的信号
    # plt.plot(filtedData)
    plt.title("acc_data")
    plt.show()
    plt.plot(gyr_data[axis_0, axis_1_start:axis_1_end, axis_2])
    plt.title("gyr_data")
    plt.show()


def yield_data(acc_data):
    p = 0
    while p + WINDOW_LENGTH < len(acc_data):
        yield acc_data[p:p + WINDOW_LENGTH]
        p = p + WINDOW_GAP


def generate_data(acc_data, gyr_data):
    acc_sequence = []
    gyr_sequence = []
    label_sequence = []
    for i in range(10):
        for data in yield_data(acc_data[i]):
            acc_sequence.append(data)
            # print("???",data.shape)
        print("acc_sequence ", i, " finished")
    print("acc_sequence finished")
    for i in range(10):
        for data in yield_data(gyr_data[i]):
            gyr_sequence.append(data)
            label_sequence.append(i)
        print("gyr_sequence ", i, " finished")
    print("gyr_sequence finished")
    return np.array(acc_sequence), np.array(gyr_sequence), np.array(label_sequence)


def get_sequence():
    if os.path.exists("acc_sequence.npy") and USE_OLD_SEQ:
        acc_sequence = np.load("acc_sequence.npy")
        gyr_sequence = np.load("gyr_sequence.npy")
        label_sequence = np.load("label_sequence.npy")
        print("Acc_seq, Gyr_seq and Label_seq loaded: ", acc_sequence.shape, gyr_sequence.shape, label_sequence.shape)
    else:
        if os.path.exists("acc_data.npy") and USE_OLD_DATA:
            acc_data = np.load("acc_data.npy")
            gyr_data = np.load("gyr_data.npy")
            print("Acc_data and Gyr_data loaded: ", acc_data.shape, gyr_data.shape)
        else:
            acc_data, gyr_data = load_data()
            np.save("acc_data.npy", acc_data)
            np.save("gyr_data.npy", gyr_data)
        acc_data = data_filter(acc_data)
        gyr_data = data_filter(gyr_data)
        print("Processed data: ", acc_data.shape, gyr_data.shape)
        plot_data(acc_data, gyr_data)
        acc_sequence, gyr_sequence, label_sequence = generate_data(acc_data, gyr_data)
        print("acc_sequence:", acc_sequence.shape, "gyr_sequence:", gyr_sequence.shape, "label_sequence:", label_sequence.shape)
        np.save("acc_sequence.npy", acc_sequence)
        np.save("gyr_sequence.npy", gyr_sequence)
        np.save("label_sequence.npy", label_sequence)
    return acc_sequence, gyr_sequence, label_sequence
