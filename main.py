import os
import re
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt


pattern = re.compile(r'SSI: (.*) SNR: (.*)\n')
rcs_data_path = 'resources/rcs'
X_data_path = 'resources/X.npy'
y_data_path = 'resources/y.npy'
regr_model_path = 'resources/regr.pkl'


def load_data_from_resources():
    fs = os.listdir(rcs_data_path)
    X, y = [], []

    for file_name in fs:
        print('Loading ' + file_name)
        with open(rcs_data_path + '/' + file_name) as fp:
            data = fp.read()
            res = pattern.findall(data)
            for pair in res:
                X.append(int(pair[0]))
                y.append(int(pair[1]))

    return np.array(X).reshape(-1, 1), np.array(y).reshape(-1, 1)


def load_data_from_npy():
    return np.load(X_data_path).astype(float), np.load(y_data_path).astype(float)


def load_model_from_pkl():
    with open(regr_model_path, "rb") as fp:
        regr = pickle.load(fp)
        return regr


def draw():
    X, y = load_data_from_npy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regr = load_model_from_pkl()
    plt.xlabel('SSI')
    plt.ylabel('SNR')
    plt.plot(X_test, regr.predict(X_test), c='k')
    plt.scatter(X_test, y_test, c='b')
    plt.show()


def main():
    X, y = load_data_from_npy()
    y_average = np.average(y)
    y_std = np.std(y)
    print("y_average:%d" % y_average)
    print("y_std:%d" % y_std)
    print("Interval:(%d,%d)" % (y_average-3*y_std, y_average+3*y_std))


if __name__ == '__main__':
    main()

