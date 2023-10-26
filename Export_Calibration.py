import pandas as pd
import numpy as np
import os

path_folder = r'D:\Luan Van\data_sensor\Calib\2023-09-21'
pathsavefile = r'D:\Luan Van\data_sensor\Calib\signal_data_calibration (25-10-2023).csv'
path = r'D:\Luan Van\data_sensor\Calib\final_data_calibration (25-10-2023).csv'


def get_Data(path):
    os.chdir(path)
    files = sorted(os.listdir(path))
    name_csv = []
    for f in files:
        if f.endswith('.csv'):
            name_csv.append(f)
    df = pd.DataFrame()
    for f in name_csv:
        data = pd.read_csv(f,
                           skiprows=[i for i in range(0, 21)])
        df = df.append(data)
    return name_csv, df


name_csv_nir, data_frame_1 = get_Data(path_folder)
name_csv_nir_calib, data_frame_2 = get_Data(path_folder)


def change_shape(data):  # ham chuyen du lieu tu cot sang hang
    file = pd.DataFrame()
    i = 0
    srange = range(0, len(data['Sample Signal (unitless)']), 228)
    for ii in srange:
        i += 228
        file = file.append(data.iloc[ii:i, 3], ignore_index=True)
    return file


def cal(datacalib, data):
    ii = 0
    values = []
    for X in datacalib.values:
        for i in range(0, 3):
            values.append(data.values[ii] / X)
            ii = ii + 1  # .....
    values_calibrated = np.array(values)
    return values_calibrated


def Calculator(calib_data, signal_data):
    tinh_pho = 0
    values_calib = []
    for A in calib_data.values:
        for B in signal_data.values:
            tinh_pho = B/A
            break
        values_calib.append([tinh_pho])
    values_calib = np.array(values_calib)
    return values_calib


name_csv1, df1 = get_Data(path_folder)
name_csv2, df2 = get_Data(path_folder + '\Calib_update')

listWavelegh = df1.values[0:228, 0].tolist()
data = df1.values[0:228, 3].tolist()
data = pd.DataFrame(data)
list_calib = df2.values[0:228, 3].tolist()

list_calib = pd.DataFrame(list_calib)

data_pho = Calculator(list_calib, data)

fileinte = change_shape(df1)
filecalibinte = change_shape(df2)

filecalibinte = np.array(filecalibinte)
final_Data = pd.DataFrame(filecalibinte, columns=listWavelegh)


list_number = []
for name in name_csv_nir:
    name = name.replace('_', '')
    if name[0] == 'e':
        list_number.append(name[1])

    if name[0] != 'e':
        list_number.append(name[1])

final_Data.to_csv(pathsavefile, index=False, header=True, na_rep='Unknown')


def f_to_list(f_variable, num_algorism):
    list1 = []
    n = 0
    variable2 = str(f_variable)
    for i in variable2:
        list1 += i
    if list1[(num_algorism - 1)] == '.':
        print('banana')
        num_algorism += 1
    list1 = list1[0:(num_algorism)]
    print(list1)


tb_data = pd.read_csv(fr'{pathsavefile}')

n = 0
list_tb_cong = []
for v in listWavelegh:
    answer = 0
    mylist = tb_data[f'{v}'].tolist()
    # print(mylist)
    len_mylist = len(mylist)
    for n in mylist:
        answer += n
    answer = answer / 30
    answer = round(answer)
    list_tb_cong.append([answer])

list_tb_cong = pd.DataFrame(list_tb_cong)
list_tb_cong = list_tb_cong.T
list_tb_cong = np.array(list_tb_cong)
list_tb_cong = pd.DataFrame(list_tb_cong, columns=listWavelegh)

list_tb_cong.to_csv(path, index=False, header=True, na_rep='Unknown')
