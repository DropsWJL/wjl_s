import os
import re
import time
import numpy as np
from numpy import mean
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt
import my_utils

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_histogram():
    # mse = [2238, 239, 2163, 241, 244]
    # r2 = [0.96406, 0.99341, 0.95831, 0.99160, 0.99092]
    fig, ax = plt.subplots()
    label = ['different ways']
    x = np.arange(len(label))
    width = 0.35
    rects1 = ax.bar(x - (2 * width) / 3, [0.96406], width / 5, label='原始数据')
    rects2 = ax.bar(x - width / 3, [0.99341], width / 5, label='去除-850以下的SSI数据')
    rects3 = ax.bar(x, [0.95831], width / 5, label='去除异常数据')
    rects4 = ax.bar(x + width / 3, [0.99160], width / 5, label='先去除-850以下的SSI数据再去除异常数据')
    rects5 = ax.bar(x + (2 * width) / 3, [0.99092], width / 5, label='先去除异常数据再去除-850以下的SSI数据')
    ax.set_ylabel('R^2')
    ax.set_title('R^2 in different ways')
    ax.set_xticks(x)
    ax.set_xticklabels(label)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)

    def autolabel(rects_list: list):
        """Attach a text label above each bar for *rects* in rects_list, displaying its height."""
        for rects in rects_list:
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom'
                            )

    autolabel([rects1, rects2, rects3, rects4, rects5])
    plt.show()


def draw_snr_ssi(snr, ssi, title):
    X_train, X_test, y_train, y_test = train_test_split(snr, ssi, test_size=0.2, random_state=0)
    train_model(X_train, X_test, y_train, y_test)
    regr = my_utils.load_model_from_pkl()

    y_predict = regr.predict(X_test)
    plt.plot(X_test, y_predict, color="blue", label="predict")
    plt.scatter(X_test, y_test, c='g', marker='o')
    plt.title(title)
    plt.xlabel('SNR')
    plt.ylabel('SSI')
    plt.xlim((0, 1200))
    plt.ylim((-1300, 200))
    plt.legend()
    plt.grid()
    plt.show()


def train_model(X_train, X_test, y_train, y_test):
    regr = linear_model.LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1)
    regr.fit(X_train, y_train)
    print('model\'s score: ' + str(regr.score(X_test, y_test)))

    with open('resources/regr.pkl', 'wb') as fp:
        pickle.dump(regr, fp)


def calculate_r2(snr, ssi):
    X_train, X_test, y_train, y_test = train_test_split(snr, ssi, test_size=0.2, random_state=0)
    train_model(X_train, X_test, y_train, y_test)
    regr = my_utils.load_model_from_pkl()

    y_predict = regr.predict(X_test)
    r2 = r2_score(y_true=y_test, y_pred=y_predict)

    return r2


def interval(data):
    data_average = np.average(data)
    data_std = np.std(data)
    print("average:%d" % data_average)
    print("std:%d" % data_std)
    print("Interval:(%d,%d)" % (data_average - 2 * data_std, data_average + 2 * data_std))
    return [data_average - 2 * data_std, data_average + 2 * data_std]


def denoise(snr, ssi, mode: str):
    """
        用于snr,ssi的数据降噪,根据@para:mode确定降噪的算法
        mode=sigma: 根据2sigma去除部分数据点
        mode=lof: 根据lof算法去除离群点
        mode=limit: 去除ssi小于-850的数据点
    """
    snr_ssi = np.concatenate((snr, ssi), axis=1)
    print("@Before snr_ssi's Shape: " + str(snr_ssi.shape))
    if mode == 'sigma':
        ssi_min, ssi_max = interval(ssi)
        snr_ssi = snr_ssi[np.all(snr_ssi >= ssi_min, axis=1), :]
        snr_ssi = snr_ssi[np.any(snr_ssi <= ssi_max, axis=1), :]
    elif mode == 'limit':
        snr_ssi = snr_ssi[np.all(snr_ssi >= -850, axis=1), :]
    elif mode == 'lof':
        print("loading lof")
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, algorithm='auto', n_jobs=-1)
        print("predicting label")
        label_predict = lof.fit_predict(snr_ssi)
        print("removing outlier")
        snr_ssi = snr_ssi[label_predict > 0, :]
    else:
        raise ValueError(mode)

    print("@After: snr_ssi's Shape: " + str(snr_ssi.shape))
    return snr_ssi[:, 0].reshape(-1, 1), snr_ssi[:, 1].reshape(-1, 1)


def main():

    draw_histogram()


if __name__ == '__main__':
    main()


