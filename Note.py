import pandas as pd
import numpy as np
import os


path_folder = r'D:\Luan Van\data_sensor\2023-09-27'
path_save_file = r'D:\Luan Van\Data\Demo_Data_270923.csv'
path_Final_Data = r'D:\Luan Van\Data\Final_Data_270923.csv'


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


folder_sensor = r'D:\Luan Van\data_sensor\2023-09-27'
name_data, data_main = get_Data(folder_sensor)

# Lấy data ở cột đầu tiên và cột thứ 4
listWavelegh = data_main.values[0:125, 0].tolist()

# chuyển cột thành hàng thông qua function change_shape
data_main = change_shape(data_main)
data_main = pd.DataFrame(data_main)
# data_calib = change_shape(calib)
data_main.drop(data_main.columns[125:], axis=1, inplace=True)
# print(data_main)

# Drop data Calib
data_calib = pd.read_csv(r'D:\Luan Van\data\Calib\final_data_calibration.csv')
data_calib = pd.DataFrame(data_calib)
data_calib.drop(data_calib.columns[125:], axis=1, inplace=True)
data_calib = data_calib.T
# print(data_calib)

data_ref = Reference(data_calib, data_main)

# Chuyển mảng thành ma trận
data_ref = np.array(data_ref)

# chuyển data sang dataframe
final_Data = pd.DataFrame(data_ref, columns=listWavelegh)

list_column = ['Ratio', 'Acid', 'Brix', 'Date', 'Point', 'Position', 'Number']

for item in list_column:
    exec(f'{item} = []')

for item in list_column:
    print(f'{item}: {eval(item)}')

# Hàm xuất excel
Cultivar = []

list_all = [Ratio, Acid, Brix, Date, Point, Position, Number]

for name in name_data:
    Cultivar = 'Quyt Duong'
    name = name.replace('_', '')
    if name[0] == 'e':
        Number.append(name[1:4])
        if name[8] == 'A':
            Position.append('Segments')
        if name[8] == 'B':
            Position.append('Mid of Segments')
        Point.append(name[6:8])
        Date.append(name[9:19])
        Brix.append(name[1])
        Acid.append(name[1])
        Ratio.append(name[1])
    if name[0] != 'e':
        Cultivar = 'Quyt Duong'
        Number.append(name[1:4])
        if name[8] == 'A':
            Position.append('Segments')
        if name[8] == 'B':
            Position.append('Mid of Segments')
        Point.append(name[6:8])
        Date.append(name[9:19])
        Brix.append(name[1])
        Acid.append(name[1])
        Ratio.append(name[1])

# print(list_time)
dem = 0
for i in list_column:
    for clib in range(dem, len(list_all)):
        final_Data.insert(loc=0, column=f'{i}', value=list_all[clib])
        break
    dem += 1
final_Data.insert(loc=0, column='Cultivar', value=Cultivar)
final_Data.to_csv(path_save_file, index=False, header=True, na_rep='Unknown')
