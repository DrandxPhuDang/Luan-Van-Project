import os
import numpy as np
import pandas as pd


class Export_Data:
    def __init__(self, file_name, path_folder_sensor, path_save, path_calib_file, list_column, Cultivar_name):
        super().__init__()
        self.path_folder = fr'{path_folder_sensor}'
        self.path_folder_save = fr'{path_save}'
        self.path_save_file = self.path_folder_save + fr'\{file_name}.csv'

        def Get_data(path):
            """
            :param path: Thư mục chứa các file csv từ cảm biến
            :return: Trả về tên file và giá trị trong file
            """
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

        def Change_shape(data):
            """
            :param data: Chuyển đổi shape của giá trị từ dọc thành ngang
            :return: trả về dataframe chứa các giá trị đã chuyển
            """
            file = pd.DataFrame()
            i = 0
            srange = range(0, len(data['Sample Signal (unitless)']), 228)
            for ii in srange:
                i += 228
                file = file.append(data.iloc[ii:i, 3], ignore_index=True)
            return file

        def Reference(calib_data, signal_data):
            """
            :param calib_data: Nhập giá trị từ file calibration (Final.... .cvs)
            :param signal_data: Nhập giá trị cột signal của file csv cảm biến
            :return: Giá trị đã tính toán với công thức là signal/calib
            """
            bien_dem = 0
            values_calib = []
            values_calib_2 = []
            for list_sig_data in signal_data.values:
                for sig_data in list_sig_data:
                    for clib in calib_data.values[bien_dem]:
                        ref_data = sig_data / clib
                        values_calib.append([ref_data])
                    bien_dem = bien_dem + 1
                bien_dem = 0
                values_calib = pd.DataFrame(values_calib).T
                values_calib_2 = pd.DataFrame(values_calib_2).append([values_calib])
                values_calib = []
            return values_calib_2

        self.name_data, self.data_main = Get_data(self.path_folder)
        self.listWavelength = self.data_main.values[0:125, 0].tolist()
        self.data_main = Change_shape(self.data_main)
        self.data_main = pd.DataFrame(self.data_main)
        self.data_main.drop(self.data_main.columns[125:], axis=1, inplace=True)
        self.data_calib = pd.read_csv(fr"{path_calib_file}")
        self.data_calib = pd.DataFrame(self.data_calib)
        self.data_calib.drop(self.data_calib.columns[125:], axis=1, inplace=True)
        self.data_calib = self.data_calib.T
        self.data_ref = Reference(self.data_calib, self.data_main)
        self.final_Data = pd.DataFrame(np.array(self.data_ref), columns=self.listWavelength)

        def export_excel():
            """
            :return: Hàm xử lí pandas chèn tên cột và các giá trị tương ứng
            """
            Cultivar = []
            Date = []
            Brix = []
            Acid = []
            Ratio = []
            Number = []
            Point = []
            Position = []
            list_all = [Ratio, Acid, Brix, Date, Point, Position, Number]

            for name in self.name_data:
                Cultivar = 'QD'
                name = name.replace('_', '')
                if name[0] == 'e':
                    Number.append(name[1:4])
                    Position.append(name[8])
                    Point.append(name[6:8])
                    Date.append(name[9:19])
                    Brix.append(name[1])
                    Acid.append(name[1])
                    Ratio.append(name[1])
                if name[0] != 'e':
                    Cultivar = 'QD'
                    Number.append(name[1:4])
                    Position.append(name[8])
                    Point.append(name[6:8])
                    Date.append(name[9:19])
                    Brix.append(name[1])
                    Acid.append(name[1])
                    Ratio.append(name[1])

            dem = 0
            for i in list_column:
                for clib in range(dem, len(list_all)):
                    self.final_Data.insert(loc=0, column=f'{i}', value=list_all[clib])
                    break
                dem += 1

            self.final_Data.insert(loc=0, column=f'{Cultivar_name}', value=Cultivar)
            self.final_Data.to_csv(self.path_save_file, index=False, header=True, na_rep='Unknown')

            print(f'Successful export data excel to {self.path_folder_save}')

        export_excel()
