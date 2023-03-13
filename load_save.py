import os
import re
import numpy as np
import time
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pickle
import json
import matplotlib.pyplot as plt

rcs_data_path = 'resources/data/rcs'
odo_data_path = 'resources/data/odo'
raw_ssi_data_path = 'resources/data/snr_ssi/raw_ssi.npy'
raw_snr_data_path = 'resources/data/snr_ssi/raw_snr.npy'
ssi_data_path = 'resources/data/snr_ssi/clean_ssi.npy'
snr_data_path = 'resources/data/snr_ssi/clean_snr.npy'
LR_model_path = 'resources/model/LR.pkl'
RANSAC_model_path = 'resources/model/RANSAC.pkl'


def load_ap_from_resources():
    fs = os.listdir(rcs_data_path)
    pattern = re.compile(r'display_aps AP(.*)_wlan\d')
    ap = []

    for file_name in fs:
        print('Loading ' + file_name)
        with open(rcs_data_path + '/' + file_name) as fp:
            data = fp.read()
            res_list = pattern.findall(data)
            for res in res_list:
                ap_name = res[0]
                if ap_name not in ap:
                    ap.append(ap_name)

    return ap


def load_snr_ssi_from_resources():
    fs = os.listdir(rcs_data_path)
    pattern = re.compile(r'SSI: (.*) SNR: (.*)\n')
    ssi, snr = [], []

    for file_name in fs:
        print('Loading ' + file_name)
        with open(rcs_data_path + '/' + file_name) as fp:
            data = fp.read()
            res_list = pattern.findall(data)
            for res in res_list:
                snr.append(int(res[0]))
                ssi.append(int(res[1]))

    return np.array(snr).reshape(-1, 1), np.array(ssi).reshape(-1, 1)


def load_snr_ssi_from_npy():
    return np.load(snr_data_path).astype(float), np.load(ssi_data_path).astype(float)


def load_data_dict_time_ap_ssi_from_sources():
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
                if int(ssi) <= -850:
                    continue

                if time_format not in dict_data.keys():
                    dict_data[time_format] = {}
                if ap not in dict_data[time_format].keys():
                    dict_data[time_format][ap] = [int(ssi)]
                else:
                    dict_data[time_format][ap].append(int(ssi))

    return dict_data


def load_data_dict_time_ap_snr_from_sources(part: int):
    fs = os.listdir(rcs_data_path)
    dict_data = {}
    pattern = re.compile(r'(\d+-\d+-\d+ \d+:\d+:\d+) .* The AP AP(.*)_wlan\d SNR: (\d+)\n')
    if part <= 4:
        fs = fs[300*(part-1):300*part]
    elif part == 5:
        fs = fs[1200:1448]
    else:
        return None

    for file_name in fs:
        print('Loading ' + file_name)
        with open(rcs_data_path + '/' + file_name) as fp:
            data = fp.read()
            res_list = pattern.findall(data)
            for res in res_list:
                time_format, ap, snr = res

                if time_format not in dict_data.keys():
                    dict_data[time_format] = {}
                if ap not in dict_data[time_format].keys():
                    dict_data[time_format][ap] = [int(snr)]
                else:
                    dict_data[time_format][ap].append(int(snr))

    return dict_data


def load_model_from_pkl(model_type: str):
    """
    :param model_type: 可选LR、RANSAC
    :return: linear_model
    """
    if model_type == 'LR':
        with open(LR_model_path, "rb") as fp:
            model = pickle.load(fp)
    elif model_type == 'RANSAC':
        with open(RANSAC_model_path, "rb") as fp:
            model = pickle.load(fp)
    else:
        model = None

    return model


def load_data_dict_time_pos_from_resources():
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


def save_in_json(data, file_name: str):
    with open(file_name, 'w', encoding='utf-8') as fp:
        str_ = json.dumps(data, ensure_ascii=False)
        fp.write(str_)


def load_from_json(file_name: str):
    with open(file_name, 'r', encoding='utf-8') as fp:
        data = fp.readline().strip()
        data_dict = json.loads(data)

    return data_dict

