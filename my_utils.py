import os
import re
import numpy as np
import time
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pickle
import json
import matplotlib.pyplot as plt

rcs_data_path = 'resources/rcs'
odo_data_path = 'resources/odo'
ssi_data_path = 'resources/raw_ssi.npy'
snr_data_path = 'resources/raw_snr.npy'
regr_model_path = 'resources/regr.pkl'
dict_time_ssi_data_path = 'resources/dict_time_ssi.pkl'
dict_time_position_data_path = 'resources/dict_time_position.pkl'


def load_rcs_data_snr_ssi_from_resources():
    fs = os.listdir(rcs_data_path)
    pattern = re.compile(r'SSI: (.*) SNR: (.*)\n')
    ssi, snr = [], []

    for file_name in fs:
        print('Loading ' + file_name)
        with open(rcs_data_path + '/' + file_name) as fp:
            data = fp.read()
            res = pattern.findall(data)
            for pair in res:
                snr.append(int(pair[0]))
                ssi.append(int(pair[1]))

    return np.array(snr).reshape(-1, 1), np.array(ssi).reshape(-1, 1)


def load_rcs_data_snr_ssi_from_npy():
    return np.load(snr_data_path).astype(float), np.load(ssi_data_path).astype(float)


def load_rcs_data_dict_time_ap_ssi_from_sources():
    fs = os.listdir(rcs_data_path)
    pattern = re.compile(r'(\d+-\d+-\d+ \d+:\d+:\d+) .* display_aps AP(.*)_wlan\d+ SSI: (.*) SNR: \d+\n')
    dict_data = {}

    for file_name in fs:
        print('Loading ' + file_name)
        with open(rcs_data_path + '/' + file_name) as fp:
            data = fp.read()
            res_list = pattern.findall(data)
            for res in res_list:
                time_format, ap, ssi = res
                if time_format not in dict_data.keys():
                    dict_data[time_format] = {}
                if ap not in dict_data[time_format].keys():
                    dict_data[time_format][ap] = [int(ssi)]
                else:
                    dict_data[time_format][ap].append(int(ssi))

    return dict_data


def load_model_from_pkl():
    with open(regr_model_path, "rb") as fp:
        regr = pickle.load(fp)
        return regr


def load_odo_data_dict_time_pos_from_resources():
    fs = os.listdir(odo_data_path)
    # fs = ['2023_0101_0101A.txt']  # for test
    pattern = re.compile(r'(\d+\.\d+) \d+ (\d) (\d+) .*\n')
    dict_time_position = {}

    for file_name in fs:
        print('Loading ' + file_name)
        with open(odo_data_path + '/' + file_name) as fp:
            data = fp.read()
            res_list = pattern.findall(data)
            for res in res_list:
                time_stamp = float(res[0])
                time_struct = time.localtime(time_stamp)
                time_format = time.strftime('%Y-%m-%d %H:%M:%S', time_struct)
                if time_format not in dict_time_position.keys():
                    dict_time_position[time_format] = (int(res[1]), int(res[2]))

    return dict_time_position


def load_odo_data_dict_time_pos_from_pkl():
    with open(dict_time_position_data_path, 'rb') as fp:
        dict_time_position = pickle.load(fp)

    return dict_time_position


def save_in_json(data_dict: dict, file_name: str):
    with open(file_name, 'w', encoding='utf-8') as fp:
        str_ = json.dumps(data_dict, ensure_ascii=False)
        fp.write(str_)


def load_from_json(file_name: str) -> dict:
    with open(file_name, 'r', encoding='utf-8') as fp:
        data = fp.readline().strip()
        data_dict = json.loads(data)

    return data_dict

