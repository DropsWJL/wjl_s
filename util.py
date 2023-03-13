import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import load_save
import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.neighbors import LocalOutlierFactor

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
    model = load_save.load_model_from_pkl(model_type)

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
    :param model_type: 线性回归预测模型的类型，可选 LR、RANSAC
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
    model = load_save.load_model_from_pkl(model_type)
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


def dict_key_str2int(data):
    res = {}
    for key in data.keys():
        res[int(key)] = data[key]

    return res

