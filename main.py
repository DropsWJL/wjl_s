import os
import re
import time
import numpy as np
from numpy import polyfit, poly1d
from numpy import mean
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt
import my_load

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

LR_mse = [2238, 239, 2163, 241, 244]
LR_r2 = [0.96406, 0.99341, 0.95831, 0.99160, 0.99092]
RANSAC_r2 = [0.96381, 0.99341, 0.95831, 0.99160, 0.99091]
date_list = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07',
             '2023-01-08', '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14',
             '2023-01-15', '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20', '2023-01-21',
             '2023-01-22', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27', '2023-01-28']

test_list = ['2023-01-01', '2023-01-02', '2023-01-03']


def draw_histogram(data_list: list):
    """
    :param data_list: 需要罗列的数据,类型为list,len=5
    :return: 无
    """
    fig, ax = plt.subplots()
    label = ['different ways']
    x = np.arange(len(label))
    width = 0.35
    rects1 = ax.bar(x - (2 * width) / 3, [data_list[0]], width / 5, label='原始数据')
    rects2 = ax.bar(x - width / 3, [data_list[1]], width / 5, label='去除-850以下的SSI数据')
    rects3 = ax.bar(x, [data_list[2]], width / 5, label='去除异常数据')
    rects4 = ax.bar(x + width / 3, [data_list[3]], width / 5, label='先去除-850以下的SSI数据再去除异常数据')
    rects5 = ax.bar(x + (2 * width) / 3, [data_list[4]], width / 5, label='先去除异常数据再去除-850以下的SSI数据')
    ax.set_ylabel('R^2')
    ax.set_title('R^2 in different ways')
    ax.set_xticks(x)
    ax.set_xticklabels(label)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)

    def autolabel(rects_list: list):
        for rects in rects_list:
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom'
                            )

    autolabel([rects1, rects2, rects3, rects4, rects5])
    plt.show()


def draw_snr_ssi(snr, ssi, model_type: str, title):
    """
    :param model_type: 线性回归预测模型的类型，可选 lr、RANSAC
    :param snr: snr数据
    :param ssi: ssi数据
    :param title: 图表的标题
    :return: 无返回值
    """
    X_train, X_test, y_train, y_test = train_test_split(snr, ssi, test_size=0.2, random_state=0)
    train_model(X_train, y_train, model_type)
    model = my_load.load_model_from_pkl(model_type)

    y_predict = model.predict(X_test)
    plt.plot(X_test, y_predict, color="blue", label="predict")
    plt.scatter(X_test, y_test, c='g', marker='o')
    plt.title(title)
    plt.xlabel('SNR')
    plt.ylabel('SSI')
    plt.legend()
    plt.grid()
    plt.show()


def train_model(X_train, y_train, model_type: str):
    """
    :param X_train: X数据
    :param y_train: y数据
    :param model_type: 线性回归预测模型的类型，可选 lr、RANSAC
    :return: 将模型写入文件中，没有返回值
    """
    if model_type == 'LR':
        model = linear_model.LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1)
    elif model_type == 'RANSAC':
        model = linear_model.RANSACRegressor()
    else:
        model = None

    model.fit(X_train, y_train)

    with open('resources/model/' + model_type + '.pkl', 'wb') as fp:
        pickle.dump(model, fp)


def calculate_r2(snr, ssi, model_type: str):
    """
    :param model_type: 线性回归预测模型的类型，可选 lr、RANSAC
    :param snr: snr数据
    :param ssi: ssi数据
    :return: 关于ssi的R2数值
    """
    X_train, X_test, y_train, y_test = train_test_split(snr, ssi, test_size=0.2, random_state=0)
    train_model(X_train, y_train, model_type)
    model = my_load.load_model_from_pkl(model_type)
    y_predict = model.predict(X_test)
    r2 = r2_score(y_true=y_test, y_pred=y_predict)

    return r2


def sigma_interval(data):
    """
    :param data: 待估计区间的数据集
    :return: 2-sigma区间
    """
    data_average = np.average(data)
    data_std = np.std(data)
    return [data_average - 2 * data_std, data_average + 2 * data_std]


def denoise(snr, ssi, mode: str):
    """
    :param snr: snr数据;
    :param ssi: ssi数据;
    :param mode: mode=sigma: 根据2sigma去除部分数据点;
                 mode=lof: 根据lof算法去除离群点;
                 mode=limit: 去除ssi小于-850的数据点;
    :return: 经过降噪的snr,ssi数据
    """
    snr_ssi = np.concatenate((snr, ssi), axis=1)
    print("@Before snr_ssi's Shape: " + str(snr_ssi.shape))
    if mode == 'sigma':
        ssi_min, ssi_max = sigma_interval(ssi)
        snr_ssi = snr_ssi[np.all(snr_ssi >= ssi_min, axis=1), :]
        snr_ssi = snr_ssi[np.any(snr_ssi <= ssi_max, axis=1), :]
    elif mode == 'limit':
        snr_ssi = snr_ssi[np.all(snr_ssi >= -850, axis=1), :]
    elif mode == 'lof':
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, algorithm='auto', n_jobs=-1)
        label_predict = lof.fit_predict(snr_ssi)
        snr_ssi = snr_ssi[label_predict > 0, :]
    else:
        raise ValueError(mode)

    print("@After: snr_ssi's Shape: " + str(snr_ssi.shape))
    return snr_ssi[:, 0].reshape(-1, 1), snr_ssi[:, 1].reshape(-1, 1)


def main():

    pos_ssi_root_path = 'resources/data/ap/pos_ssi/'
    ap = '0001'
    pos_ssi_list = []

    for test_date in test_list:
        temp_list = my_load.load_from_json(pos_ssi_root_path + test_date + '/' + ap + '.json')
        pos_ssi_list.extend(temp_list)

    pos_ssi_list.sort(key=lambda x: x[0])
    pos = [x[0] for x in pos_ssi_list]
    ssi = [x[1] for x in pos_ssi_list]
    pos = np.array(pos) / 5000
    ssi = np.array(ssi)
    ssi_pred = poly1d(polyfit(pos, ssi, 48))
    plt.plot(pos.reshape(-1, 1), ssi_pred(pos), c='red', linewidth=3.0)
    plt.scatter(pos.reshape(-1, 1), ssi.reshape(-1, 1), c='blue')
    plt.xlabel('POS')
    plt.ylabel('SSI')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

    # 根据 dict_time_pos 以及 time_ssi数据，将pos_ssi数据以天数进行分组
    # dict_time_pos = my_load.load_from_json('resources/data/dict_time_position.json')
    # time_ssi_root_path = 'resources/data/ap/time_ssi/'
    # pos_ssi_root_path = 'resources/data/ap/pos_ssi/'
    # for date in date_list:
    #     print('[loading] date: ' + date + ', ' + 'time: ' + time.strftime('%Y-%m-%d %H:%M:%S',
    #                                                                       time.localtime(time.time())))
    #     json_file_list = os.listdir(time_ssi_root_path + date)
    #     for json_file in json_file_list:
    #         print('[parsing] file: ' + json_file + ', ' + 'time: ' + time.strftime('%Y-%m-%d %H:%M:%S',
    #                                                                                time.localtime(time.time())))
    #         time_ssi_list = my_load.load_from_json(time_ssi_root_path + date + '/' + json_file)
    #         ap_data_dict = {}
    #         for time_ssi in time_ssi_list:
    #             time_stamp, ssi = time_ssi
    #             if time_stamp not in dict_time_pos:
    #                 continue
    #             pos = dict_time_pos[time_stamp][1]
    #             if pos not in ap_data_dict:
    #                 ap_data_dict[pos] = [ssi]
    #             else:
    #                 ap_data_dict[pos].append(ssi)
    #
    #             ap_data_list = []
    #             for pos in ap_data_dict.keys():
    #                 ssi = mean(ap_data_dict[pos])
    #                 ap_data_list.append((pos, ssi))
    #
    #             ap_data = sorted(ap_data_list, key=lambda x: x[0])
    #             if not os.path.isdir(pos_ssi_root_path + date):
    #                 os.makedirs(pos_ssi_root_path + date)
    #             my_load.save_in_json(ap_data, pos_ssi_root_path + date + '/' + json_file)

    # 根据全量time_ssi数据，以时间为标记进行分组
    # root_path = 'resources/data/ap/time_ssi/all'
    # json_file_list = os.listdir(root_path)
    #
    # for json_file in json_file_list:
    #     print('parsing ' + json_file)
    #     time_ssi_list = my_load.load_from_json(root_path + '/' + json_file)
    #     for date in date_list:
    #         date_time_ssi_list = list(filter(lambda x: date in x[0], time_ssi_list))
    #         if not os.path.isdir('resources/data/ap/time_ssi/' + date):
    #             os.makedirs('resources/data/ap/time_ssi/' + date)
    #
    #         my_load.save_in_json(date_time_ssi_list, 'resources/data/ap/time_ssi/' + date + '/' + json_file)

    # 为一个ap绘制(位置，强度)图，数据为全量
    # pos_ssi_list = my_load.load_from_json('resources/data/ap/pos_ssi/all/0001.json')
    # pos = [x[0] for x in pos_ssi_list]
    # ssi = [x[1] for x in pos_ssi_list]
    # pos = np.array(pos).reshape(-1, 1)
    # ssi = np.array(ssi).reshape(-1, 1)
    #
    # poly_reg = PolynomialFeatures(degree=20)
    # pos_ploy = poly_reg.fit_transform(pos)
    # lin_reg = linear_model.Lasso(fit_intercept=True)
    # lin_reg.fit(pos_ploy, ssi)
    # ssi_pred = lin_reg.predict(pos_ploy)
    #
    # plt.plot(pos, ssi_pred, c='black')
    # plt.scatter(pos, ssi, c='green')
    # plt.xlabel('POS')
    # plt.ylabel('SSI')
    # plt.grid()
    # plt.show()

    # 从time_ssi数据中，借助dict_time_position数据，提取出pos_ssi数据
    # ap_time_ssi_path = 'resources/data/ap/time_ssi'
    # ap_pos_ssi_path = 'resources/data/ap/pos_ssi'
    # ap_json_list = os.listdir(ap_time_ssi_path)
    # dict_time_pos = my_load.load_from_json('resources/data/dict_time_position.json')
    #
    # for json_file in ap_json_list:
    #     print('parsing ' + json_file)
    #     # 先保存为字典，可以汇聚多个pos下的ssi值
    #     ap_data_dict = {}
    #     time_ssi_list = my_load.load_from_json(ap_time_ssi_path + '/' + json_file)
    #     for time_ssi in time_ssi_list:
    #         time_stamp, ssi = time_ssi
    #         if time_stamp not in dict_time_pos:
    #             continue
    #
    #         pos = dict_time_pos[time_stamp][1]
    #         if pos not in ap_data_dict:
    #             ap_data_dict[pos] = [ssi]
    #         else:
    #             ap_data_dict[pos].append(ssi)
    #
    #     ap_data_list = []
    #     for pos in ap_data_dict.keys():
    #         ssi = mean(ap_data_dict[pos])
    #         ap_data_list.append((pos, ssi))
    #
    #     ap_data = sorted(ap_data_list, key=lambda x: x[0])
    #     my_load.save_in_json(ap_data, ap_pos_ssi_path + '/' + json_file)

    # 从原始数据中提取ap的time_ssi数据
    # ap_list = my_utils.load_from_json('resources/ap_list.json')
    # dict_time_ap_ssi = my_utils.load_from_json('resources/dict_time_ap_ssi.json')
    #
    # for ap in ap_list:
    #     ap_data = []
    #     for time_stamp in dict_time_ap_ssi.keys():
    #         time_stamp_value = dict_time_ap_ssi[time_stamp]
    #         if ap in time_stamp_value:
    #             ap_data.append((time_stamp, mean(time_stamp_value[ap])))
    #
    #     my_utils.save_in_json(ap_data, 'resources/ap/time_ssi/' + ap + '.json')

