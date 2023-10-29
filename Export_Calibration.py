import pandas as pd
import numpy as np
import os


class Export_calib:
    def __init__(self, path_folder, name_file):
        super().__init__()

        def get_Data(path):
            """
            :param path: Thư mục chứa các file csv từ cảm biến
            :return: Trả về tên file và giá trị trong file
            """
            os.chdir(path)
            files = sorted(os.listdir(path))
            name = []
            for f in files:
                if f.endswith('.csv'):
                    name.append(f)
            data = pd.DataFrame()
            for f in name:
                data_name = pd.read_csv(f, skiprows=[i for i in range(0, 21)])
                data = data.append(data_name)
            return name, data

        def change_shape(data_change):
            """
            :param data_change: Chuyển đổi shape của giá trị từ dọc thành ngang
            :return: trả về dataframe chứa các giá trị đã chuyển
            """
            file = pd.DataFrame()
            i = 0
            srange = range(0, len(data_change['Sample Signal (unitless)']), 228)
            for ii in srange:
                i += 228
                file = file.append(data_change.iloc[ii:i, 3], ignore_index=True)
            return file

        name_csv, df = get_Data(path_folder + r'\Calib_update')
        listWavelength = df.values[0:228, 0].tolist()
        final_Data = pd.DataFrame(np.array(change_shape(df)), columns=listWavelength)

        list_number = []
        for call_name in name_csv:
            call_name = call_name.replace('_', '')
            if call_name[0] == 'e':
                list_number.append(call_name[1])
            if call_name[0] != 'e':
                list_number.append(call_name[1])

        final_Data.to_csv(path_folder + r'\signal_data_calibration ' + name_file + '.csv',
                          index=False, header=True, na_rep='Unknown')

        tb_data = pd.read_csv(path_folder + r'\signal_data_calibration ' + name_file + '.csv')

        list_tb_cong = []
        for v in listWavelength:
            answer = 0
            mylist = tb_data[f'{v}'].tolist()
            for n in mylist:
                answer += n
            answer = answer / 30
            answer = round(answer)
            list_tb_cong.append([answer])

        list_tb_cong = np.array(pd.DataFrame(list_tb_cong).T)
        list_tb_cong = pd.DataFrame(list_tb_cong, columns=listWavelength)

        list_tb_cong.to_csv(path_folder + r'\final_data_calibration ' + name_file + '.csv',
                            index=False, header=True, na_rep='Unknown')
