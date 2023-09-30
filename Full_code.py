import math
import os
from kennard_stone import train_test_split
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestRegressor
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mlxtend.preprocessing import MeanCenterer
from sklearn import preprocessing


class data_excel:

    def __init__(self, file_name, path_folder_sensor, path_save, path_calib_file, list_column, Cultivar_name):
        # tao bien dung chung
        super().__init__()
        self.path_folder = fr'{path_folder_sensor}'
        self.path_folder_save = fr'{path_save}'
        self.path_save_file = self.path_folder_save + fr'\{file_name}.csv'

        def Get_data(path):
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

        # get data from csv sensor
        self.name_data, self.data_main = Get_data(self.path_folder)

        # get data first column và cột thứ 4
        self.listWavelength = self.data_main.values[0:126, 0].tolist()

        # column to row function change_shape
        self.data_main = Change_shape(self.data_main)
        self.data_main = pd.DataFrame(self.data_main)

        # data_calib = change_shape(calib)
        self.data_main.drop(self.data_main.columns[126:], axis=1, inplace=True)

        # Drop data Calib
        self.data_calib = pd.read_csv(fr"{path_calib_file}")
        self.data_calib = pd.DataFrame(self.data_calib)
        self.data_calib.drop(self.data_calib.columns[126:], axis=1, inplace=True)
        self.data_calib = self.data_calib.T

        # Tinh gia tri reference
        self.data_ref = Reference(self.data_calib, self.data_main)

        # List to array
        self.data_ref = np.array(self.data_ref)

        # data to dataframe
        self.final_Data = pd.DataFrame(self.data_ref, columns=self.listWavelength)

        def export_excel():
            # export excel

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
                Cultivar = 'Quyt Duong'
                name = name.replace('_', '')
                if name[0] == 'e':
                    Number.append(name[1:4])
                    if name[8] == 'A':
                        Position.append('Mid of Segments')
                    if name[8] == 'B':
                        Position.append('Mid of 2 Segments')
                    Point.append(name[6:8])
                    Date.append(name[9:19])
                    Brix.append(name[1])
                    Acid.append(name[1])
                    Ratio.append(name[1])
                if name[0] != 'e':
                    Cultivar = 'Quyt Duong'
                    Number.append(name[1:4])
                    if name[8] == 'A':
                        Position.append('Mid of Segments')
                    if name[8] == 'B':
                        Position.append('Mid of 2 Segments')
                    Point.append(name[6:8])
                    Date.append(name[9:19])
                    Brix.append(name[1])
                    Acid.append(name[1])
                    Ratio.append(name[1])

            # Export Data Excel
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


class data_find_best_waves:

    def __init__(self, k_train, n_estimators, namefile, path_file_data, path_save):
        # tao bien dung chung
        super().__init__()

        self.Radar_array_cuff = pd.DataFrame()
        self.Radar = []
        self.R_train = []
        self.R_test = []
        self.ob_score = []
        warnings.filterwarnings("ignore")
        self.list_tb_cong = []
        self.clf_rand = RandomForestRegressor(n_estimators=n_estimators, oob_score=True)
        print(f'Finding Important Waves...')

        def np2pd(XX):
            df = pd.DataFrame()
            XX = XX.reshape(-1, 1)
            self.ii = 0
            self.a = []
            for self.i in XX:
                self.a = str(self.ii)
                df[self.a] = XX[self.ii, :]
                self.ii = self.ii + 1
                if self.ii == XX.shape[0]:
                    self.ii = XX.shape[0] - 1
            return df

        def value_train(data):

            for i in range(0, data):
                if i == 10:
                    print("Lines trained", i)
                if i == 25:
                    print("Lines trained", i)
                if i == 50:
                    print("Lines trained", i)
                if i == 75:
                    print("Lines trained", i)
                if i == 100:
                    print("Lines trained", i)
                if i > 100:
                    print("Lines trained over 100, not safe")

                self.clf_rand.fit(self.X_cal, self.y_cal)
                radar = np2pd(self.clf_rand.feature_importances_)

                products_list = radar.values
                products_list.tolist()

                # cal mean of data waves
                for v in products_list:
                    answer = 0
                    len_mylist = len(v)
                    for n in v:
                        answer += n
                    answer = answer / len_mylist
                    self.list_tb_cong.append([answer])

                self.Radar_array_cuff = self.Radar_array_cuff.append([radar], ignore_index=True)
            return self.Radar_array_cuff

        def change_value(list_mean, list_value):
            stt = 0
            compare = []
            compare_data = []
            for mean in list_mean:
                mean = mean[0]
                for lst in range(stt, len(list_value)):
                    c_list = list_value[lst]
                    for val_c in c_list:
                        if mean > val_c:
                            ss = 0
                            compare.append(ss)
                        if mean < val_c:
                            ss = 1
                            compare.append(ss)
                    compare_data.append(compare)
                    compare = []
                    break
                stt += 1
            compare_data = np.array(compare_data)
            self.df_compare_data = pd.DataFrame(compare_data, columns=self.feature_list)
            self.path_3 = path_save + fr'\{namefile}' + '_file_nhi_phan' + '.csv'
            self.df_compare_data.to_csv(fr'{self.path_3}', index=False,
                                        header=True, na_rep='Unknown')
            return compare_data

        def math_waves_important(list_data, data, number_run):
            sum_val_X = []
            loc_sum_val_X = []
            val_BC = []
            loc_val_BC = []
            for wave_bc in list_data:
                val_X = 0
                for X in data[wave_bc]:
                    val_X += X
                sum_val_X.append([val_X])
                val_BC.append([wave_bc])
                if val_X > number_run:
                    loc_sum_val_X.append(val_X)
                    loc_val_BC.append(wave_bc)
            return sum_val_X, val_BC, loc_sum_val_X, loc_val_BC

        # choose file data
        self.data_preprocess = pd.read_csv(fr"{path_file_data}")

        # get data from waves
        self.y_Train = self.data_preprocess.Brix.values
        # get data from waves
        self.X_Train = self.data_preprocess.iloc[:, 12:].values
        # get data from waves
        self.feature_list = list(self.data_preprocess.iloc[:, 12:].columns)
        self.X_cal = self.X_Train
        self.y_cal = self.y_Train.ravel()

        # data train to list, value_train(input number train)
        self.list_Radar_array_cuff = value_train(k_train).values
        self.list_Radar_array_cuff.tolist()

        # call function convert data doi ve 0, 1
        self.check_01 = change_value(self.list_tb_cong, self.list_Radar_array_cuff)

        # list to array
        self.check_01 = np.array(self.check_01)
        self.check_01 = pd.DataFrame(self.check_01)

        # list to array
        self.data_ref = np.array(self.Radar_array_cuff)
        self.final_Data = pd.DataFrame(self.data_ref, columns=self.feature_list)

        # list to array
        self.list_tb_cong = pd.DataFrame(self.list_tb_cong)
        self.list_tb_cong = np.array(self.list_tb_cong)
        self.list_tb_cong = pd.DataFrame(self.list_tb_cong)

        # add waves value 0, 1
        jj = len(self.feature_list)
        self.list_re_waves = []
        while jj > 0:
            jj = jj - 1
            val = self.check_01[jj]
            bc = self.feature_list[jj]
            bc = 'w: ' + bc
            self.list_re_waves.append(bc)
            self.final_Data.insert(loc=len(self.feature_list), column=bc, value=val)

        self.final_Data.insert(loc=len(self.feature_list), column='Mean', value=self.list_tb_cong)

        # Goi ham tinh
        self.find_im = math_waves_important(self.list_re_waves, self.final_Data, number_run=round(k_train / 2))

        # get part waves
        self.im_val = self.find_im[3]
        self.len_bc = (len(self.im_val) - 1)

        # except data nan
        self.list_val_bc = []
        while self.len_bc > 0:
            self.val_bc = self.im_val[self.len_bc]
            self.val_bc = self.val_bc[3:]
            self.list_val_bc.append(self.val_bc)
            self.len_bc = self.len_bc - 1

        # change shape
        self.list_val_bc = np.array(self.list_val_bc)
        self.data_im_val = pd.DataFrame(self.list_val_bc).T

        def save_waves_important():
            self.path_1 = path_save + fr'\{namefile}' + '_file_val' + '.csv'
            self.path_2 = path_save + fr'\{namefile}' + '_file_waves' + '.csv'
            self.data_im_val.to_csv(fr"{self.path_2}", index=False,
                                    header=False, na_rep='Unknown')
            # print(final_Data)
            self.final_Data.to_csv(fr"{self.path_1}", index=False,
                                   header=True, na_rep='Unknown')

            print(f'Successful export data excel to {path_save}')

        save_waves_important()


class data_train_test:

    def __init__(self, path_file_data, path_file_new_waves, full_or_part, path_save, test_size):
        # tao bien dung chung
        super().__init__()

        print(f'Splitting Data Train_Test...')

        self.df = pd.read_csv(path_file_data)
        self.listWavelength = self.df.iloc[:0, 12:]
        full_waves = [f'{e}' for e in self.listWavelength]

        self.listW = pd.read_csv(path_file_new_waves)
        part_waves = [f'{e}' for e in self.listW]

        if full_or_part == 'full':
            self.data_final = self.df[full_waves].values
            self.wavelength = full_waves
        if full_or_part == 'part':
            self.data_final = self.df[part_waves].values
            self.wavelength = part_waves

        # Chia data X: val_wavelengths, y: Brix
        self.X = self.data_final
        self.y = self.df['Brix']

        # train test split
        self.X_train_all, self.X_test_all, self.y_train_all, self.y_test_all = train_test_split(self.X, self.y,
                                                                                                test_size=test_size)

        self.X_train_all = np.array(self.X_train_all)
        self.X_test_all = np.array(self.X_test_all)
        self.X_train_all = pd.DataFrame(self.X_train_all, columns=self.wavelength)
        self.X_test_all = pd.DataFrame(self.X_test_all, columns=self.wavelength)

        self.y_train_all = np.array(self.y_train_all)
        self.y_test_all = np.array(self.y_test_all)
        self.y_train_all = pd.DataFrame(self.y_train_all)
        self.y_test_all = pd.DataFrame(self.y_test_all)

        self.X_train_all.insert(loc=0, column='Brix', value=self.y_train_all)
        self.X_test_all.insert(loc=0, column='Brix', value=self.y_test_all)

        print(f'Successful export data non_preprocess excel to {path_save}')

        self.X_train_all.to_csv(path_save + r'\non_prepro_train' + '.csv', index=False, header=True)
        self.X_test_all.to_csv(path_save + r'\non_prepro_test' + '.csv', index=False, header=True)


class data_preprocess:

    def __init__(self, path_train_test):
        # tao bien dung chung
        super().__init__()

        def msc(input_data):
            input_data = np.array(input_data, dtype=np.float64)
            ref = []
            sampleCount = int(len(input_data))
            # mean centre correction
            for i in range(input_data.shape[0]):
                input_data[i, :] -= input_data[i, :].mean()
            data_msc = np.zeros_like(input_data)
            for i in range(input_data.shape[0]):
                for j in range(0, sampleCount, 10):
                    ref.append(np.mean(input_data[j:j + 10], axis=0))
                    fit = np.polyfit(ref[i], input_data[i, :], 1, full=True)
                    data_msc[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]
            return data_msc

        self.data_train = pd.read_csv(f'{path_train_test}' + r'\non_prepro_train' + '.csv')
        self.data_test = pd.read_csv(f'{path_train_test}' + r'\non_prepro_test' + '.csv')

        self.X_train = self.data_train.iloc[:, 1:]
        self.y_train = self.data_train['Brix']
        self.X_test = self.data_test.iloc[:, 1:]
        self.y_test = self.data_test['Brix']

        wavelengths_re = self.data_train.iloc[:0, 1:]
        full_waves = [f'{e}' for e in wavelengths_re]

        # Using preprocess on X_TRAIN
        def prepro_X_train():
            self.X_train = preprocessing.normalize(self.X_train.values)
            prepro_normal_train = preprocessing.Normalizer().fit(self.X_train)
            self.X_train = prepro_normal_train.transform(self.X_train)
            self.X_train = savgol_filter(self.X_train, window_length=5, polyorder=2)
            scaler_X_train = preprocessing.MinMaxScaler(feature_range=(0, 2))
            self.X_train = scaler_X_train.fit_transform(self.X_train)
            self.X_train = scaler_X_train.inverse_transform(self.X_train)
            self.X_train = msc(self.X_train)

        prepro_X_train()

        # Using preprocess on X_TEST
        def prepro_X_test():
            self.X_test = preprocessing.normalize(self.X_test.values)
            prepro_normal_test = preprocessing.Normalizer().fit(self.X_test)
            self.X_test = prepro_normal_test.transform(self.X_test)
            self.X_test = savgol_filter(self.X_test, window_length=5, polyorder=2)
            scaler_X_test = preprocessing.MinMaxScaler(feature_range=(0, 2))
            self.X_test = scaler_X_test.fit_transform(self.X_test)
            self.X_test = scaler_X_test.inverse_transform(self.X_test)
            self.X_test = msc(self.X_test)

        prepro_X_test()

        # Mean_Center
        mc_pls = MeanCenterer().fit(self.X_train)
        self.X_train = mc_pls.transform(self.X_train)
        self.X_test = mc_pls.transform(self.X_test)

        # Add thanh data frame
        self.X_train = pd.DataFrame(self.X_train, columns=full_waves)
        self.X_train_pre_all = pd.DataFrame(self.X_train)
        # Add thanh data frame
        self.X_test = pd.DataFrame(self.X_test, columns=full_waves)
        self.X_test_pre_all = pd.DataFrame(self.X_test)

        self.X_train_pre_all.insert(loc=0, column='Brix', value=self.y_train)
        self.X_test_pre_all.insert(loc=0, column='Brix', value=self.y_test)

        print(f'Successful export data preprocess excel to {path_train_test}')

        self.X_train_pre_all.to_csv(path_train_test + r'\prepro_train' + '.csv', index=False, header=True)
        self.X_test_pre_all.to_csv(path_train_test + r'\prepro_test' + '.csv', index=False, header=True)


class data_predict_regression:

    def __init__(self, model_regression, path_file_data, preprocess):
        # tao bien dung chung
        super().__init__()
        if preprocess == 'pre':
            self.data_pre_train = pd.read_csv(f'{path_file_data}' + r'\prepro_train' + '.csv')
            self.data_pre_test = pd.read_csv(f'{path_file_data}' + r'\prepro_test' + '.csv')
        if preprocess == 'non':
            self.data_pre_train = pd.read_csv(f'{path_file_data}' + r'\non_prepro_train' + '.csv')
            self.data_pre_test = pd.read_csv(f'{path_file_data}' + r'\non_prepro_test' + '.csv')

        self.X_pre_train = self.data_pre_train.iloc[:, 1:]
        self.y_pre_train = self.data_pre_train['Brix']
        self.X_pre_test = self.data_pre_test.iloc[:, 1:]
        self.y_pre_test = self.data_pre_test['Brix']

        self.X_pre_all = self.X_pre_train.append(self.X_pre_test)
        self.y_pre_all = self.y_pre_train.append(self.y_pre_test)

        if model_regression == 'PLS':
            name_model_pls = 'PLS'
            # choose number component fit by Cross-validation
            param = {
                'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                 25]}
            model_pls = PLSRegression()
            search = GridSearchCV(model_pls, param, cv=10, scoring='neg_mean_squared_error', return_train_score=True,
                                  refit=True)
            search.fit(self.X_pre_train, self.y_pre_train)
            # Train model cross-validation
            model_pls = PLSRegression(n_components=search.best_params_["n_components"])

            model_pls.fit(self.X_pre_train, self.y_pre_train)
            # Evaluate model
            y_train_pred_pls = model_pls.predict(self.X_pre_train)
            y_test_pred_pls = model_pls.predict(self.X_pre_test)
            # R, R_Squared, R_MSE
            print('--------------- TRAIN--------------------')
            R_Train_pls = np.corrcoef(self.y_pre_train, y_train_pred_pls, rowvar=False)
            print('R:', "{:.3f}".format(R_Train_pls[0][1]))
            R_Squared_Train_pls = r2_score(self.y_pre_train, y_train_pred_pls)
            print('R^2:', "{:.3f}".format(R_Squared_Train_pls))
            print(f"Accuracy: {R_Squared_Train_pls * 100:.3f}%")
            RMSE_Train_pls = math.sqrt(mean_squared_error(self.y_pre_train, y_train_pred_pls))
            print('R_MSE :', "{:.3f}".format(RMSE_Train_pls))
            MAE_Train_pls = mean_absolute_error(self.y_pre_train, y_train_pred_pls)
            print('MAE:', "{:.3f}".format(MAE_Train_pls))
            # R, R_Squared, R_MSE
            print('--------------- TEST--------------------')
            R_Test_pls = np.corrcoef(self.y_pre_test, y_test_pred_pls, rowvar=False)
            print('R:', "{:.3f}".format(R_Test_pls[0][1]))
            R_Squared_Test_pls = r2_score(self.y_pre_test, y_test_pred_pls)
            print('R^2:', "{:.3f}".format(R_Squared_Test_pls))
            print(f"Accuracy: {R_Squared_Test_pls * 100:.3f}%")
            RMSE_Test_pls = math.sqrt(mean_squared_error(self.y_pre_test, y_test_pred_pls))
            print('R_MSE :', "{:.3f}".format(RMSE_Test_pls))
            MAE_Test_pls = mean_absolute_error(self.y_pre_test, y_test_pred_pls)
            print('MAE:', "{:.3f}".format(MAE_Test_pls))
            print('--------------- RPD--------------------')
            RPD_Test_pls = np.std(self.y_pre_test) / RMSE_Test_pls
            print('RPD:', "{:.2f}".format(RPD_Test_pls))

            def load_spectrum():
                plt.scatter(self.y_pre_test, y_test_pred_pls)
                plt.xlabel('Actual Response')
                plt.ylabel('Predicted Response')
                plt.title(f'{name_model_pls} Regression (R²={R_Squared_Test_pls:.2f})')
                reg_pls = np.polyfit(self.y_pre_test, y_test_pred_pls, deg=1)
                trend_pls = np.polyval(reg_pls, self.y_pre_test)
                plt.plot(self.y_pre_test, trend_pls, 'r')
                plt.show()

            load_spectrum()

        if model_regression == 'RF':
            name_model_rf = 'Random Forest'

            model_rf = RandomForestRegressor()

            # Define the hyperparameters for grid search
            param_grid = {
                'n_estimators': [10, 50, 100],
                'max_features': ['auto', 'sqrt', 'log2'],
                'min_samples_split': [2, 4, 8],
                'bootstrap': [True, False]
            }

            # Perform grid search to find the best hyperparameters
            grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=5)
            grid_search.fit(self.X_pre_train, self.y_pre_train)

            y_train_pred_rf = grid_search.predict(self.X_pre_train)
            y_test_pred_rf = grid_search.predict(self.X_pre_test)

            # R, R_Squared, R_MSE
            print('--------------- TRAIN--------------------')
            R_Train_rf = np.corrcoef(self.y_pre_train, y_train_pred_rf, rowvar=False)
            print('R:', "{:.3f}".format(R_Train_rf[0][1]))
            R_Squared_Train_rf = r2_score(self.y_pre_train, y_train_pred_rf)
            print('R^2:', "{:.3f}".format(R_Squared_Train_rf))
            print(f"Accuracy: {R_Squared_Train_rf * 100:.3f}%")
            RMSE_Train_rf = math.sqrt(mean_squared_error(self.y_pre_train, y_train_pred_rf))
            print('R_MSE :', "{:.3f}".format(RMSE_Train_rf))
            MAE_Train_rf = mean_absolute_error(self.y_pre_train, y_train_pred_rf)
            print('MAE:', "{:.3f}".format(MAE_Train_rf))
            # R, R_Squared, R_MSE
            print('--------------- TEST--------------------')
            R_Test_rf = np.corrcoef(self.y_pre_test, y_test_pred_rf, rowvar=False)
            print('R:', "{:.3f}".format(R_Test_rf[0][1]))
            R_Squared_Test_rf = r2_score(self.y_pre_test, y_test_pred_rf)
            print('R^2:', "{:.3f}".format(R_Squared_Test_rf))
            print(f"Accuracy: {R_Squared_Test_rf * 100:.3f}%")
            RMSE_Test_rf = math.sqrt(mean_squared_error(self.y_pre_test, y_test_pred_rf))
            print('R_MSE :', "{:.3f}".format(RMSE_Test_rf))
            MAE_Test_rf = mean_absolute_error(self.y_pre_test, y_test_pred_rf)
            print('MAE:', "{:.3f}".format(MAE_Test_rf))
            print('--------------- RPD--------------------')
            RPD_Test_rf = np.std(self.y_pre_test) / RMSE_Test_rf
            print('RPD:', "{:.2f}".format(RPD_Test_rf))

            def load_spectrum():
                plt.scatter(self.y_pre_test, y_test_pred_rf)
                plt.xlabel('Actual Response')
                plt.ylabel('Predicted Response')
                plt.title(f'{name_model_rf} Regression (R²={R_Squared_Test_rf:.2f})')
                reg_rf = np.polyfit(self.y_pre_test, y_test_pred_rf, deg=1)
                trend_rf = np.polyval(reg_rf, self.y_pre_test)
                plt.plot(self.y_pre_test, trend_rf, 'r')
                plt.show()

            load_spectrum()

        if model_regression == 'L':
            name_model_l = 'Linear'

            model_l = LinearRegression()
            model_l.fit(self.X_pre_train, self.y_pre_train)
            y_train_pred_l = model_l.predict(self.X_pre_train)
            y_test_pred_l = model_l.predict(self.X_pre_test)

            # R, R_Squared, R_MSE
            print('--------------- TRAIN--------------------')
            R_Train_l = np.corrcoef(self.y_pre_train, y_train_pred_l, rowvar=False)
            print('R:', "{:.3f}".format(R_Train_l[0][1]))
            R_Squared_Train_l = round(r2_score(self.y_pre_train, y_train_pred_l))
            print('R^2:', "{:.3f}".format(R_Squared_Train_l))
            print(f"Accuracy: {R_Squared_Train_l * 100:.3f}%")
            RMSE_Train_l = round(math.sqrt(mean_squared_error(self.y_pre_train, y_train_pred_l)))
            print('R_MSE :', "{:.3f}".format(RMSE_Train_l))
            MAE_Train_l = round(mean_absolute_error(self.y_pre_train, y_train_pred_l))
            print('MAE:', "{:.3f}".format(MAE_Train_l))
            # R, R_Squared, R_MSE
            print('--------------- TEST--------------------')
            R_Test_l = np.corrcoef(self.y_pre_test, y_test_pred_l, rowvar=False)
            print('R:', "{:.3f}".format(R_Test_l[0][1]))
            R_Squared_Test_l = round(r2_score(self.y_pre_test, y_test_pred_l))
            print('R^2:', "{:.3f}".format(R_Squared_Test_l))
            print(f"Accuracy: {R_Squared_Test_l * 100:.2f}%")
            RMSE_Test_l = round(math.sqrt(mean_squared_error(self.y_pre_test, y_test_pred_l)))
            print('R_MSE :', "{:.3f}".format(RMSE_Test_l))
            MAE_Test_l = round(mean_absolute_error(self.y_pre_test, y_test_pred_l))
            print('MAE:', "{:.3f}".format(MAE_Test_l))
            print('--------------- RPD--------------------')
            RPD_Test_l = round(np.std(self.y_pre_test) / RMSE_Test_l)
            print('RPD:', "{:.2f}".format(RPD_Test_l))

            def load_spectrum():
                plt.scatter(self.y_pre_test, y_test_pred_l)
                plt.xlabel('Actual Response')
                plt.ylabel('Predicted Response')
                plt.title(f'{name_model_l} Regression (R²={R_Squared_Test_l:.2f})')
                reg_l = np.polyfit(self.y_pre_test, y_test_pred_l, deg=1)
                trend_l = np.polyval(reg_l, self.y_pre_test)
                plt.plot(self.y_pre_test, trend_l, 'r')
                plt.show()

            load_spectrum()


class data_spectrum:

    def __init__(self, path_file_data, name_folder, name_column, non_prepro, path_spectrum, list_data):
        # tao bien dung chung
        super().__init__()

        def msc(input_data):
            input_data = np.array(input_data, dtype=np.float64)
            ref = []
            sampleCount = int(len(input_data))
            # mean centre correction
            for i in range(input_data.shape[0]):
                input_data[i, :] -= input_data[i, :].mean()
            data_msc = np.zeros_like(input_data)
            for i in range(input_data.shape[0]):
                for j in range(0, sampleCount, 10):
                    ref.append(np.mean(input_data[j:j + 10], axis=0))
                    fit = np.polyfit(ref[i], input_data[i, :], 1, full=True)
                    data_msc[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]
            return data_msc

        def plot(ax, data, start, title):
            y = data.values[:, start - 1:]
            row, _ = y.shape  # so hang cua bo du lieu
            str_x = data.columns.values[start - 1:]
            fmx = []
            for x in str_x:
                fmx.append(float(x))
            for i in range(0, row):
                ax.plot(fmx, y[i])
            ax.set_xticks(np.arange(int(min(fmx)), int(max(fmx)), 100))
            ax.set_title(title)
            ax.grid()

        def plot_mean(data, label, start):
            y = data.values[:, start - 1:]
            y_mean = np.mean(y, axis=0)
            s_trx = data.columns.values[start - 1:]
            fmx = []
            for x in s_trx:
                fmx.append(float(x))
            plt.plot(fmx, y_mean, label=label)
            ax = plt.gca()
            ax.set_xlabel('Wavelength(nm)', fontsize=15, fontweight='bold')
            ax.set_ylabel('Intensity(a.u)', fontsize=15, fontweight='bold')

        # input column starting waves
        read_data = pd.read_csv(path_file_data + fr'\{name_folder}.csv', sep=',')
        df = pd.DataFrame(read_data)
        wavelengths_prepro_spectrum = df.iloc[:0, 12:]
        full_waves_spectrum = [f'{e}' for e in wavelengths_prepro_spectrum]
        for check in range(0, len(list_data)):
            try:
                list_data_value0 = df[df[f'{name_column}'] == f'{list_data[check]}']
                list_data_value0 = list_data_value0.iloc[:, 12:]
                prepro_list_data_value0 = msc(list_data_value0)
                list_data_value0 = pd.DataFrame(list_data_value0, columns=full_waves_spectrum)
                prepro_list_data_value0 = pd.DataFrame(prepro_list_data_value0, columns=full_waves_spectrum)
                list_data_value0.to_csv(path_spectrum + r'\non_prepro' + fr'\{list_data[check]}' + '.csv', index=False,
                                        header=True)
                prepro_list_data_value0.to_csv(path_spectrum + r'\prepro' + fr'\{list_data[check]}' + '.csv',
                                               index=False,
                                               header=True)
            except:
                pass

        print(f'Successful export data non_prepro_spectrum excel to {path_spectrum}')

        # input column starting waves
        start_col = 1
        count = 0
        if non_prepro == 'non':

            fig, Ax = plt.subplots(2, 2, figsize=(14, 7))
            fig.suptitle('Spectrum', fontsize=19, fontweight='bold')
            plt.subplots_adjust(left=0.076, right=0.96)
            fig.supxlabel('Wavelength(nm)', fontsize=15, fontweight='bold')
            fig.supylabel('Intensity(a.u)', fontsize=15, fontweight='bold')
            plt.figure(figsize=(10, 7))
            plt.title('Mean Spectrum', fontsize=19, fontweight='bold')

            for check in range(0, len(list_data)):
                try:
                    if check < 2:
                        # load file
                        Data_plot = pd.read_csv(path_spectrum + r'\non_prepro' + fr'\{list_data[check]}.csv', sep=',')
                        Data_plot = pd.DataFrame(Data_plot)
                        # load plot data
                        plot(Ax[0, check], Data_plot, start=start_col, title=f'{list_data[check]}')
                        # load plot mean data
                        plot_mean(Data_plot, start=start_col, label=f'{list_data[check]}')
                    if check >= 2:
                        # load file
                        Data_plot = pd.read_csv(path_spectrum + r'\non_prepro' + fr'\{list_data[check]}.csv', sep=',')
                        Data_plot = pd.DataFrame(Data_plot)
                        # load plot data
                        plot(Ax[1, count], Data_plot, start=start_col, title=f'{list_data[check]}')
                        # load plot mean data
                        plot_mean(Data_plot, start=start_col, label=f'{list_data[check]}')
                        count += 1
                except:
                    print('Not have data', {list_data[check]})

            plt.grid(1)
            plt.legend()
            plt.show()
            # plt.savefig(path_spectrum + r'\non_prepro_spectrum.png')
            # plt.savefig(path_spectrum + r'\non_prepro_mean_spectrum.png')

        if non_prepro == 'pre':

            fig, Ax = plt.subplots(2, 2, figsize=(14, 7))
            fig.suptitle('Spectrum', fontsize=19, fontweight='bold')
            plt.subplots_adjust(left=0.076, right=0.96)
            fig.supxlabel('Wavelength(nm)', fontsize=15, fontweight='bold')
            fig.supylabel('Intensity(a.u)', fontsize=15, fontweight='bold')
            plt.figure(figsize=(10, 7))
            plt.title('Mean Spectrum', fontsize=19, fontweight='bold')

            for check in range(0, len(list_data)):
                if check < 2:
                    Data_plot = pd.read_csv(path_spectrum + r'\prepro' + fr'\{list_data[check]}.csv', sep=',')
                    Data_plot = pd.DataFrame(Data_plot)
                    # load plot data
                    plot(Ax[0, check], Data_plot, start=start_col, title=f'{list_data[check]}')
                    # load plot mean data
                    plot_mean(Data_plot, start=start_col, label=f'{list_data[check]}')

            plt.legend()
            plt.grid(1)
            plt.show()
            plt.savefig(path_spectrum + r'\non_prepro_mean_spectrum.png')
            plt.savefig(path_spectrum + r'\prepro_spectrum.png')


# chay chuong trinh
if __name__ == "__main__":

    # path calib
    path_calib = r'D:\Luan Van\Data\Calib\final_data_calibration.csv'
    # path read file csv and path save file csv
    folder_sensor = r'D:\Luan Van\data_sensor\2023-09-27'
    path_folder_save = r'D:\Luan Van\Data\Demo_Data'

    # path file part waves
    path_save_file_loc_waves = r'D:\Luan Van\Data\loc_waves'
    path_file_loc_waves = r'D:\Luan Van\Data\loc_waves\change_val_file_waves.csv'

    # path data excel
    path_file = r'D:\Luan Van\Data\Final_Data\Data_ALL.csv'
    path_file_spectrum = r'D:\Luan Van\Data\Final_Data'

    # path save
    path_save_train_test = r'D:\Luan Van\Data\train_test'

    # path save
    path_save_prepro_spectrum = r'D:\Luan Van\Data\spectrum'

    # --------------------------------------EXPORT DATA-----------------------------------------------------------------
    data_excel(file_name='New_Demo_Data_270923',
               path_folder_sensor=folder_sensor,
               path_save=path_folder_save,
               path_calib_file=path_calib, list_column=['Ratio', 'Acid', 'Brix', 'Date',
                                                        'Point', 'Position', 'Number'],
               Cultivar_name='QD')

    # --------------------------------------FIND BEST WAVES-------------------------------------------------------------
    # data_find_best_waves(n_estimators=100,
    #                      k_train=100,
    #                      namefile='change_val',
    #                      path_file_data=path_file,
    #                      path_save=path_save_file_loc_waves)

    # --------------------------------------REGRESSION DATA-------------------------------------------------------------
    # data_train_test(path_file_data=path_file, test_size=0.3,
    #                 path_file_new_waves=path_file_loc_waves,
    #                 full_or_part='full',
    #                 path_save=path_save_train_test)
    #
    # data_preprocess(path_train_test=path_save_train_test)
    #
    # data_predict_regression(model_regression='PLS',
    #                         path_file_data=path_save_train_test,
    #                         preprocess='non')

    # --------------------------------------SPECTRUM PLOT---------------------------------------------------------------
    # data_spectrum(name_folder='Data_All',
    #               path_file_data=path_file_spectrum,
    #               path_spectrum=path_save_prepro_spectrum,
    #               name_column='Position',
    #               list_data=['Segments', 'Mid of Segments'],
    #               non_prepro='non')
