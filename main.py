import os
import re
import time
import numpy as np
from numpy import mean
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import matplotlib.pyplot as plt
import my_utils

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_histogram():
    # mse = [2238, 239, 2163, 241, 244]
    fig, ax = plt.subplots()
    label = ['different ways']
    x = np.arange(len(label))
    width = 0.35
    rects1 = ax.bar(x - (2 * width) / 3, [2238], width / 5, label='原始数据')
    rects2 = ax.bar(x - width / 3, [239], width / 5, label='去除-850以下的SSI数据')
    rects3 = ax.bar(x, [2163], width / 5, label='去除异常数据')
    rects4 = ax.bar(x + width / 3, [241], width / 5, label='先去除-850以下的SSI数据再去除异常数据')
    rects5 = ax.bar(x + (2 * width) / 3, [244], width / 5, label='先去除异常数据再去除-850以下的SSI数据')
    ax.set_ylabel('MSE')
    ax.set_title('MSE in different ways')
    ax.set_xticks(x)
    ax.set_xticklabels(label)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom'
                        )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    plt.show()


def draw_snr_ssi(snr, ssi, title):
    X_train, X_test, y_train, y_test = train_test_split(snr, ssi, test_size=0.2, random_state=0)
    train_model(X_train, X_test, y_train, y_test)
    regr = my_utils.load_model_from_pkl()

    y_predict = regr.predict(X_test)
    mse = mean_squared_error(y_test, y_predict)
    print('MSE:%d' % mse)
    plt.plot(X_test, y_predict, color="blue", label="predict")
    plt.scatter(X_test, y_test, c='g', marker='o')
    plt.title(title)
    plt.xlabel('SNR')
    plt.ylabel('SSI')
    plt.legend()
    plt.grid()
    plt.show()


def train_model(X_train, X_test, y_train, y_test):
    regr = linear_model.LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1)
    regr.fit(X_train, y_train)
    print('model\'s score: ' + str(regr.score(X_test, y_test)))

    with open('resources/regr.pkl', 'wb') as fp:
        pickle.dump(regr, fp)


def interval(data):
    data_average = np.average(data)
    data_std = np.std(data)
    print("average:%d" % data_average)
    print("std:%d" % data_std)
    print("Interval:(%d,%d)" % (data_average - 2 * data_std, data_average + 2 * data_std))
    return [data_average - 2 * data_std, data_average + 2 * data_std]


def denoise(snr, ssi):
    ssi_min, ssi_max = interval(ssi)
    snr_ssi = np.concatenate((snr, ssi), axis=1)
    print(snr_ssi.shape)
    snr_ssi = snr_ssi[np.all(snr_ssi >= ssi_min, axis=1), :]
    snr_ssi = snr_ssi[np.any(snr_ssi <= ssi_max, axis=1), :]
    print(snr_ssi.shape)

    return snr_ssi[:, 0].reshape(-1, 1), snr_ssi[:, 1].reshape(-1, 1)


def remove_outliers(snr, ssi):
    snr_ssi = np.concatenate((snr, ssi), axis=1)
    print(snr_ssi.shape)
    snr_ssi = snr_ssi[np.all(snr_ssi >= -850, axis=1), :]
    print(snr_ssi.shape)

    return snr_ssi[:, 0].reshape(-1, 1), snr_ssi[:, 1].reshape(-1, 1)


def main():
    # snr, ssi = np.load('resources/raw_denoise_snr.npy'), np.load('resources/raw_denoise_ssi.npy')
    # snr, ssi = remove_outliers(snr, ssi)
    # np.save('resources/denoise_clean_snr.npy', snr)
    # np.save('resources/denoise_clean_ssi.npy', ssi)
    #
    # snr, ssi = np.load('resources/denoise_clean_snr.npy'), np.load('resources/denoise_clean_ssi.npy')
    # draw_snr_ssi(snr, ssi, '去除了SSI小于-850的经过噪声消除的数据 SNR-SSI')
    draw_histogram()


if __name__ == '__main__':
    main()


