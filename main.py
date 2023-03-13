import os
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
import load_save
import util


LR_mse = [2238, 239, 2163, 241, 244]
LR_r2 = [0.96406, 0.99341, 0.95831, 0.99160, 0.99092]
RANSAC_r2 = [0.96381, 0.99341, 0.95831, 0.99160, 0.99091]
date_list = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07',
             '2023-01-08', '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14',
             '2023-01-15', '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20', '2023-01-21',
             '2023-01-22', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27', '2023-01-28']
test_list = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
tick_list = [1672502400, 1672588800, 1672675200, 1672761600, 1672848000, 1672934400, 1673020800,
             1673107200, 1673193600, 1673280000, 1673366400, 1673452800, 1673539200, 1673625600,
             1673712000, 1673798400, 1673884800, 1673971200, 1674057600, 1674144000, 1674230400,
             1674316800, 1674403200, 1674489600, 1674576000, 1674662400, 1674748800, 1674835200,
             1674921600]


def main():
    pos_snr2ssi_root_path = 'resources/data/ap/pos_snr2ssi/Bias/'
    pos_ssi_root_path = 'resources/data/ap/pos_ssi/'
    ap = '0001'
    pos_snr2ssi_list = []

    for test_date in test_list:
        temp_list = load_save.load_from_json(pos_snr2ssi_root_path + test_date + '/' + ap + '.json')
        pos_snr2ssi_list.extend(temp_list)
        temp_list = load_save.load_from_json(pos_ssi_root_path + test_date + '/' + ap + '.json')
        pos_snr2ssi_list.extend(temp_list)

    pos_snr2ssi_list.sort(key=lambda x: x[0])
    pos = [x[0] for x in pos_snr2ssi_list]
    ssi = [x[1] for x in pos_snr2ssi_list]
    pos = np.array(pos) / 1000
    ssi = np.array(ssi)
    ssi_pred = poly1d(polyfit(pos, ssi, 36))
    plt.plot(pos.reshape(-1, 1), ssi_pred(pos), c='red', linewidth=5.0)
    plt.scatter(pos.reshape(-1, 1), ssi.reshape(-1, 1), c='blue')
    plt.xlabel('POS')
    plt.ylabel('SSI')
    plt.title(ap)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

    # 为AP绘制(位置,强度)图
    # pos_snr2ssi_root_path = 'resources/data/ap/pos_snr2ssi/'
    # ap = '0112'
    # pos_snr2ssi_list = []
    #
    # for test_date in test_list:
    #     temp_list = load_save.load_from_json(pos_snr2ssi_root_path + test_date + '/' + ap + '.json')
    #     pos_snr2ssi_list.extend(temp_list)
    #
    # pos_snr2ssi_list.sort(key=lambda x: x[0])
    # pos = [x[0] for x in pos_snr2ssi_list]
    # ssi = [x[1] for x in pos_snr2ssi_list]
    # pos = np.array(pos) / 3000
    # ssi = np.array(ssi)
    # ssi_pred = poly1d(polyfit(pos, ssi, 48))
    # plt.plot(pos.reshape(-1, 1), ssi_pred(pos), c='red', linewidth=5.0)
    # plt.scatter(pos.reshape(-1, 1), ssi.reshape(-1, 1), c='blue')
    # plt.xlabel('POS')
    # plt.ylabel('SSI')
    # plt.grid()
    # plt.show()

    # 根据全量tick_snr2ssi数据，以日期为标记进行分组
    # root_path = 'resources/data/ap/tick_snr2ssi/all/'
    # json_file_list = os.listdir(root_path)
    #
    # for json_file in json_file_list:
    #     print('parsing ' + json_file)
    #     tick_snr2ssi_list = io.load_from_json(root_path + json_file)
    #     for index in range(0, 28):
    #         date_tick_snr2ssi_list = list(filter(
    #             lambda x: (x[0] > tick_list[index]) & (x[0] < tick_list[index + 1]),
    #             tick_snr2ssi_list))
    #
    #         if not os.path.isdir('resources/data/ap/tick_snr2ssi/' + date_list[index]):
    #             os.makedirs('resources/data/ap/tick_snr2ssi/' + date_list[index])
    #
    #         date_tick_snr2ssi_list.sort(key=lambda x: x[0])
    #         io.save_in_json(date_tick_snr2ssi_list, 'resources/data/ap/tick_snr2ssi/' + date_list[index] + '/' + json_file)

    # 收集全量pos_snr2ssi数据
    # dict_tick_pos = io.load_from_json('resources/data/tick_key/dict_tick_position.json')
    # dict_tick_pos = util.dict_key_str2int(dict_tick_pos)
    # root_path = 'resources/data/ap/tick_snr2ssi/'
    # json_file_list = os.listdir(root_path)
    #
    # for json_file in json_file_list:
    #     print('parsing ' + json_file)
    #     tick_snr2ssi = io.load_from_json(root_path + json_file)
    #     dict_pos_snr2ssi = {}
    #     for meta in tick_snr2ssi:
    #         tick, snr2ssi = meta
    #         if tick not in dict_tick_pos:
    #             continue
    #         pos = dict_tick_pos[tick][1]
    #         if pos not in dict_pos_snr2ssi:
    #             dict_pos_snr2ssi[pos] = [snr2ssi]
    #         else:
    #             dict_pos_snr2ssi[pos].append(snr2ssi)
    #
    #     list_pos_snr2ssi = [(x, round(mean(dict_pos_snr2ssi[x], 1))) for x in dict_pos_snr2ssi.keys()]
    #     list_pos_snr2ssi = sorted(list_pos_snr2ssi, key=lambda x: x[0])
    #     io.save_in_json(list_pos_snr2ssi, 'resources/data/ap/pos_snr2ssi/' + json_file)

    # 将pos_snr2ssi按照日期分组 Bias
    # dict_tick_pos = load_save.load_from_json('resources/data/tick_key/dict_tick_position.json')
    # dict_tick_pos = util.dict_key_str2int(dict_tick_pos)
    # tick_snr2ssi_root_path = 'resources/data/ap/tick_snr2ssi/'
    # pos_snr2ssi_root_path = 'resources/data/ap/pos_snr2ssi/Bias/'
    #
    # for index in range(0, 28):
    #     date = date_list[index]
    #     print('[loading] date: ' + date + ', ' + 'time: ' + time.strftime('%Y-%m-%d %H:%M:%S',
    #                                                                       time.localtime(time.time())))
    #     json_file_list = os.listdir(tick_snr2ssi_root_path + date)
    #     for json_file in json_file_list:
    #         print('[parsing] file: ' + json_file + ', ' + 'time: ' + time.strftime('%Y-%m-%d %H:%M:%S',
    #                                                                                time.localtime(time.time())))
    #         tick_snr2ssi_list = load_save.load_from_json(tick_snr2ssi_root_path + date + '/' + json_file)
    #         ap_data_dict = {}
    #         for tick_snr2ssi in tick_snr2ssi_list:
    #             tick, snr2ssi = tick_snr2ssi
    #             '''由于可能会有一到两秒的误差，这里采取近似处理，且舍去误差超过两秒的数据'''
    #             if tick in dict_tick_pos:
    #                 pos = dict_tick_pos[tick][1]
    #             elif tick+1 in dict_tick_pos:
    #                 pos = dict_tick_pos[tick+1][1]
    #             elif tick+2 in dict_tick_pos:
    #                 pos = dict_tick_pos[tick+2][1]
    #             elif tick-1 in dict_tick_pos:
    #                 pos = dict_tick_pos[tick-1][1]
    #             elif tick-2 in dict_tick_pos:
    #                 pos = dict_tick_pos[tick-2][1]
    #             else:
    #                 continue
    #
    #             if pos not in ap_data_dict:
    #                 ap_data_dict[pos] = [snr2ssi]
    #             else:
    #                 ap_data_dict[pos].append(snr2ssi)
    #
    #         ap_data_list = [(pos, mean(ap_data_dict[pos])) for pos in ap_data_dict.keys()]
    #         ap_data_list = list(filter(lambda x: x[1] > -850, ap_data_list))
    #         ap_data_list.sort(key=lambda x: x[0])
    #         if not os.path.isdir(pos_snr2ssi_root_path + date):
    #             os.makedirs(pos_snr2ssi_root_path + date)
    #         load_save.save_in_json(ap_data_list, pos_snr2ssi_root_path + date + '/' + json_file)

    # 将pos_snr2ssi按照日期分组 NoBias
    # dict_tick_pos = load_save.load_from_json('resources/data/tick_key/dict_tick_position.json')
    # dict_tick_pos = util.dict_key_str2int(dict_tick_pos)
    # tick_snr2ssi_root_path = 'resources/data/ap/tick_snr2ssi/'
    # pos_snr2ssi_root_path = 'resources/data/ap/pos_snr2ssi/'
    #
    # for index in range(0, 28):
    #     date = date_list[index]
    #     print('[loading] date: ' + date + ', ' + 'time: ' + time.strftime('%Y-%m-%d %H:%M:%S',
    #                                                                       time.localtime(time.time())))
    #     json_file_list = os.listdir(tick_snr2ssi_root_path + date)
    #     for json_file in json_file_list:
    #         print('[parsing] file: ' + json_file + ', ' + 'time: ' + time.strftime('%Y-%m-%d %H:%M:%S',
    #                                                                                time.localtime(time.time())))
    #         tick_snr2ssi_list = load_save.load_from_json(tick_snr2ssi_root_path + date + '/' + json_file)
    #         ap_data_dict = {}
    #         for tick_snr2ssi in tick_snr2ssi_list:
    #             tick, snr2ssi = tick_snr2ssi
    #             if tick not in dict_tick_pos:
    #                 continue
    #             pos = dict_tick_pos[tick][1]
    #             if pos not in ap_data_dict:
    #                 ap_data_dict[pos] = [snr2ssi]
    #             else:
    #                 ap_data_dict[pos].append(snr2ssi)
    #
    #         ap_data_list = [(pos, mean(ap_data_dict[pos])) for pos in ap_data_dict.keys()]
    #         ap_data_list = list(filter(lambda x: x[1] > -850, ap_data_list))
    #         ap_data_list.sort(key=lambda x: x[0])
    #         if not os.path.isdir(pos_snr2ssi_root_path + date):
    #             os.makedirs(pos_snr2ssi_root_path + date)
    #         load_save.save_in_json(ap_data_list, pos_snr2ssi_root_path + date + '/' + json_file)

    # 根据 dict_time_pos 以及 time_ssi数据，将pos_ssi数据以天数进行分组
    # dict_time_pos = io.load_from_json('resources/data/time_key/dict_time_position.json')
    # time_ssi_root_path = 'resources/data/ap/time_ssi/'
    # pos_ssi_root_path = 'resources/data/ap/pos_ssi/'
    # for date in date_list:
    #     print('[loading] date: ' + date + ', ' + 'time: ' + time.strftime('%Y-%m-%d %H:%M:%S',
    #                                                                       time.localtime(time.time())))
    #     json_file_list = os.listdir(time_ssi_root_path + date)
    #     for json_file in json_file_list:
    #         print('[parsing] file: ' + json_file + ', ' + 'time: ' + time.strftime('%Y-%m-%d %H:%M:%S',
    #                                                                                time.localtime(time.time())))
    #         time_ssi_list = io.load_from_json(time_ssi_root_path + date + '/' + json_file)
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
    #         ap_data_list = [(pos, mean(ap_data_dict[pos])) for pos in ap_data_dict.keys()]
    #         ap_data = sorted(ap_data_list, key=lambda x: x[0])
    #         if not os.path.isdir(pos_ssi_root_path + date):
    #             os.makedirs(pos_ssi_root_path + date)
    #         io.save_in_json(ap_data, pos_ssi_root_path + date + '/' + json_file)

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

    # 收集tick_snr
    # ap_list = io.load_from_json('resources/data/ap/ap_list.json')
    #
    # dict_tick_ap_snr_1 = io.load_from_json('resources/data/tick_key/dict_tick_ap_snr_1.json')
    # dict_tick_ap_snr_1 = util.dict_key_str2int(dict_tick_ap_snr_1)
    # dict_tick_ap_snr_2 = io.load_from_json('resources/data/tick_key/dict_tick_ap_snr_2.json')
    # dict_tick_ap_snr_2 = util.dict_key_str2int(dict_tick_ap_snr_2)
    # print("************[STAGE ONE]************")
    # for ap in ap_list:
    #     print("STAGE ONE: parsing " + ap)
    #     ap_data = []
    #     for tick in dict_tick_ap_snr_1.keys():
    #         if ap in dict_tick_ap_snr_1[tick]:
    #             ap_data.append((tick, mean(dict_tick_ap_snr_1[tick][ap])))
    #
    #     for tick in dict_tick_ap_snr_2.keys():
    #         if ap in dict_tick_ap_snr_2[tick]:
    #             ap_data.append((tick, mean(dict_tick_ap_snr_2[tick][ap])))
    #
    #     io.save_in_json(ap_data, 'resources/data/ap/tick_snr/' + ap + '.json')
    #
    # del dict_tick_ap_snr_1
    # del dict_tick_ap_snr_2
    #
    # dict_tick_ap_snr_3 = io.load_from_json('resources/data/tick_key/dict_tick_ap_snr_3.json')
    # dict_tick_ap_snr_3 = util.dict_key_str2int(dict_tick_ap_snr_3)
    # dict_tick_ap_snr_4 = io.load_from_json('resources/data/tick_key/dict_tick_ap_snr_4.json')
    # dict_tick_ap_snr_4 = util.dict_key_str2int(dict_tick_ap_snr_4)
    # print("************[STAGE TWO]************")
    # for ap in ap_list:
    #     print("STAGE TWO: parsing " + ap)
    #     ap_data = io.load_from_json('resources/data/ap/tick_snr/' + ap + '.json')
    #     for tick in dict_tick_ap_snr_3.keys():
    #         if ap in dict_tick_ap_snr_3[tick]:
    #             ap_data.append((tick, mean(dict_tick_ap_snr_3[tick][ap])))
    #
    #     for tick in dict_tick_ap_snr_4.keys():
    #         if ap in dict_tick_ap_snr_4[tick]:
    #             ap_data.append((tick, mean(dict_tick_ap_snr_4[tick][ap])))
    #
    #     io.save_in_json(ap_data, 'resources/data/ap/tick_snr/' + ap + '.json')
    #
    # del dict_tick_ap_snr_3
    # del dict_tick_ap_snr_4
    #
    # dict_tick_ap_snr_5 = io.load_from_json('resources/data/tick_key/dict_tick_ap_snr_5.json')
    # dict_tick_ap_snr_5 = util.dict_key_str2int(dict_tick_ap_snr_5)
    # print("************[STAGE THREE]************")
    # for ap in ap_list:
    #     print("STAGE THREE: parsing " + ap)
    #     ap_data = io.load_from_json('resources/data/ap/tick_snr/' + ap + '.json')
    #     for tick in dict_tick_ap_snr_5.keys():
    #         if ap in dict_tick_ap_snr_5[tick]:
    #             ap_data.append((tick, mean(dict_tick_ap_snr_5[tick][ap])))
    #
    #     io.save_in_json(ap_data, 'resources/data/ap/tick_snr/' + ap + '.json')

    # tick_snr 预测 tick_snr2ssi
    # root_path = 'resources/data/ap/tick_snr/all/'
    # lr = io.load_model_from_pkl('LR')
    # json_file_list = os.listdir(root_path)
    #
    # for json_file in json_file_list:
    #     print('parsing ' + json_file)
    #     tick_snr = io.load_from_json(root_path + json_file)
    #     tick = [x[0] for x in tick_snr]
    #     snr = [x[1] for x in tick_snr]
    #     snr2ssi = lr.predict(np.array(snr).reshape(-1, 1)).flatten().tolist()
    #     snr2ssi = [round(x, 1) for x in snr2ssi]
    #     tick_snr2ssi = [list(x) for x in zip(tick, snr2ssi)]
    #     io.save_in_json(tick_snr2ssi, 'resources/data/ap/tick_snr2ssi/' + json_file)
