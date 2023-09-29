import time

import pandas as pd
import numpy as np
import os

path_folder = r'D:\Luan Van Tot Nghiep\data_sensor\2023-09-08'
path_save_file = r'D:\Luan Van Tot Nghiep\Data\Demo_Data_080923.csv'
path_Final_Data = r'D:\Luan Van Tot Nghiep\Data\Final_Data_080923.csv'


def get_Data(path):
    os.chdir(path)
    files = sorted(os.listdir(path))
    name_csv = []
    for f in files:
        if f.endswith('.csv'):
            name_csv.append(f)
    df = pd.DataFrame()
    for f in name_csv:
        data = pd.read_csv(f, skiprows=[i for i in range(0, 21)])
        df = df.append(data)
    return name_csv, df


def change_shape(data):  # ham chuyen du lieu tu cot sang hang
    file = pd.DataFrame()
    i = 0
    srange = range(0, len(data['Sample Signal (unitless)']), 228)
    for ii in srange:
        i += 228
        file = file.append(data.iloc[ii:i, 3], ignore_index=True)
    return file


def Reference(calib_data, signal_data):
    bien_dem = 0
    values_calib = []
    values_calib_2 = []
    values_calib_3 = []
    for list_sig_data in signal_data.values:
        for sig_data in list_sig_data:
            for clib in calib_data.values[bien_dem]:
                ref_data = sig_data / clib
                values_calib.append([ref_data])
                # print(ref_data)
            bien_dem = bien_dem + 1
        bien_dem = 0

        values_calib = pd.DataFrame(values_calib)
        values_calib = values_calib.T
        values_calib_2 = pd.DataFrame(values_calib_2)
        values_calib_2 = values_calib_2.append([values_calib])
        values_calib = []
    return values_calib_2


# Lấy data từ file cảm biến
name_data, data_main = get_Data(path_folder)

# Lấy data ở cột đầu tiên và cột thứ 4
listWavelegh = data_main.values[0:125, 0].tolist()

# chuyển cột thành hàng thông qua function change_shape
data_main = change_shape(data_main)
data_main = pd.DataFrame(data_main)
# data_calib = change_shape(calib)
data_main.drop(data_main.columns[125:], axis=1, inplace=True)
# print(data_main)

# Drop data Calib
data_calib = pd.read_csv(r'D:\Luan Van Tot Nghiep\data_sensor\Calib\final_data_calibration.csv')
data_calib = pd.DataFrame(data_calib)
data_calib.drop(data_calib.columns[125:], axis=1, inplace=True)
data_calib = data_calib.T
#print(data_calib)

data_ref = Reference(data_calib, data_main)

# Chuyển mảng thành ma trận
data_ref = np.array(data_ref)

# chuyển data sang dataframe
final_Data = pd.DataFrame(data_ref, columns=listWavelegh)

# Hàm xuất excel
list_name = []
list_position = []
list_time = []
list_brix = []
list_acid = []
list_ratio = []
list_number = []
list_side = []

for name in name_data:
    name = name.replace('_', '')
    if name[0] == 'e':
        list_name = 'Quyt Duong'
        list_number.append(name[1:4])
        list_side.append(name[5:8])
        list_time.append(name[8:18])
        list_brix.append(name[1])
        list_acid.append(name[1])
        list_ratio.append(name[1])
    if name[0] != 'e':
        list_name = 'Quyt Duong'
        list_number.append(name[1:4])
        list_side.append(name[5:8])
        list_time.append(name[8:18])
        list_brix.append(name[1])
        list_acid.append(name[1])
        list_ratio.append(name[1])

# print(list_time)
final_Data.insert(loc=0, column='Ratio Brix/Acid', value=list_ratio)
final_Data.insert(loc=0, column='Acid', value=list_acid)
final_Data.insert(loc=0, column='Brix', value=list_brix)
final_Data.insert(loc=0, column='Date of Expriment', value=list_time)
final_Data.insert(loc=0, column='Cultivar', value=list_name)
final_Data.insert(loc=0, column='Point Measurements', value=list_side)
final_Data.insert(loc=0, column='Number of Tangerine', value=list_number)

final_Data.to_csv(path_save_file, index=False, header=True, na_rep='Unknown')
