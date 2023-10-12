import math
import os
from kennard_stone import train_test_split
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import warnings

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


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
        self.listWavelength = self.data_main.values[0:228, 0].tolist()

        # column to row function change_shape
        self.data_main = Change_shape(self.data_main)
        self.data_main = pd.DataFrame(self.data_main)

        # data_calib = change_shape(calib)
        self.data_main.drop(self.data_main.columns[228:], axis=1, inplace=True)

        # Drop data Calib
        self.data_calib = pd.read_csv(fr"{path_calib_file}")
        self.data_calib = pd.DataFrame(self.data_calib)
        self.data_calib.drop(self.data_calib.columns[228:], axis=1, inplace=True)
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
                Cultivar = 'QD'
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
                    Cultivar = 'QD'
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


class Preprocessing_data:

    def __init__(self, path_file_data, path_file_new_waves, full_or_part, path_save):
        # tao bien dung chung
        super().__init__()

        print(f'Splitting Data Train_Test...')
        try:
            self.df = pd.read_csv(path_file_data)
            self.listWavelength = self.df.iloc[:0, 12:]
            self.full_waves = [f'{e}' for e in self.listWavelength]

            self.listW = pd.read_csv(path_file_new_waves)
            self.part_waves = [f'{e}' for e in self.listW]
        except:
            pass

        if full_or_part == 'full':
            self.data_final = self.df[self.full_waves].values
            self.wavelength = self.full_waves
        if full_or_part == 'part':
            self.data_final = self.df[self.part_waves].values
            self.wavelength = self.part_waves

        # Chia data X: val_wavelengths, y: Brix
        self.X = self.data_final
        self.y = self.df['Brix']

        pd.DataFrame(self.X).to_csv(path_save + r'\X_non_prepro_data' + '.csv', index=False, header=True)
        pd.DataFrame(self.y).to_csv(path_save + r'\Y_non_prepro_data' + '.csv', index=False, header=True)

        print(f'Successful export data_non_preprocessing excel to {path_save}')

        def mean_center(X):
            mean_X = np.mean(X)
            X_centered = X - mean_X
            return X_centered

        def remove_outliers(data_X, data_y, threshold):
            # Create and fit the PLS regression model
            regressor = PLSRegression(n_components=4)
            regressor.fit(data_X, data_y)
            # Predict the target values
            y_pred = regressor.predict(data_X)
            # Calculate the residuals
            residuals = data_y - y_pred
            # Set the threshold for outliers
            threshold_ = threshold * np.std(residuals)
            # Set the threshold for outliers
            outliers_mask = np.abs(residuals) > threshold_
            # Set the threshold for outliers
            X_clean_data = pd.DataFrame(data_X[~outliers_mask])
            y_clean_data = pd.DataFrame(data_y[~outliers_mask])
            return X_clean_data, y_clean_data

        def preprocessing_data(X_data):
            X_data = pd.DataFrame(X_data).dropna()
            # Fill in missing values using mean imputation
            X_data.fillna(X_data.mean(), inplace=True)
            # normalize
            X_data = preprocessing.normalize(X_data.values)
            # Normalizer
            prepro_normal_train = preprocessing.Normalizer().fit(X_data)
            X_data = prepro_normal_train.transform(X_data)
            # S_Filter
            X_data = savgol_filter(X_data, window_length=5, polyorder=2)
            # Min Max Scaler
            scaler_X_train = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            X_data = scaler_X_train.fit_transform(X_data)
            X_data = scaler_X_train.inverse_transform(X_data)
            X_data = pd.DataFrame(X_data)
            return X_data

        # X_clean = mean_center(self.X)
        X_clean = preprocessing_data(self.X)
        # X_clean, y_clean = remove_outliers(X_clean, self.y, threshold=1)

        X_clean.to_csv(path_save + r'\X_prepro_data' + '.csv', index=False, header=True)
        pd.DataFrame(self.y).to_csv(path_save + r'\Y_prepro_data' + '.csv', index=False, header=True)

        print(f'Successful export data_preprocessing excel to {path_save}')


class split_data:

    def __init__(self, path_train_test, test_size, prepro_or_none):
        # tao bien dung chung
        super().__init__()
        if prepro_or_none == 'None':
            self.data_X = pd.read_csv(f'{path_train_test}' + r'\X_non_prepro_data' + '.csv')
            self.data_Y = pd.read_csv(f'{path_train_test}' + r'\Y_non_prepro_data' + '.csv')
        if prepro_or_none == 'Prepro':
            self.data_X = pd.read_csv(f'{path_train_test}' + r'\X_prepro_data' + '.csv')
            self.data_Y = pd.read_csv(f'{path_train_test}' + r'\Y_prepro_data' + '.csv')

        wavelengths_re = self.data_X.iloc[:0, 0:]
        full_waves = [f'{e}' for e in wavelengths_re]

        X_train, X_test, y_train, y_test = train_test_split(self.data_X, self.data_Y, test_size=test_size)

        # Add thanh data frame
        X_train = pd.DataFrame(X_train, columns=full_waves)
        X_train_pre_all = pd.DataFrame(X_train)
        # Add thanh data frame
        X_test = pd.DataFrame(X_test, columns=full_waves)
        X_test_pre_all = pd.DataFrame(X_test)
        X_train_pre_all.insert(loc=0, column='Brix', value=y_train)
        X_test_pre_all.insert(loc=0, column='Brix', value=y_test)
        X_train_pre_all.to_csv(path_train_test + r'\data_train' + '.csv', index=False, header=True)
        X_test_pre_all.to_csv(path_train_test + r'\data_test' + '.csv', index=False, header=True)


class data_predict_regression:

    def __init__(self, model_regression, path_file_data):
        # tao bien dung chung
        super().__init__()

        self.data_pre_train = pd.read_csv(f'{path_file_data}' + r'\data_train' + '.csv')
        self.data_pre_test = pd.read_csv(f'{path_file_data}' + r'\data_test' + '.csv')

        self.X_pre_train = self.data_pre_train.iloc[:, 1:]
        self.y_pre_train = self.data_pre_train['Brix']
        self.X_pre_test = self.data_pre_test.iloc[:, 1:]
        self.y_pre_test = self.data_pre_test['Brix']

        self.all_X = pd.DataFrame(self.X_pre_train.append(self.X_pre_test))
        self.all_y = pd.DataFrame(self.y_pre_train.append(self.y_pre_test))

        if model_regression == 'PLS':
            name_model_pls = 'PLS'

            model = PLSRegression(n_components=5)
            # model = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=0.1)
            # model = RandomForestRegressor(n_estimators=59,
            #                               random_state=42, max_depth=15)

            model.fit(self.X_pre_train, self.y_pre_train)

            # Evaluate model
            y_train_pred_pls = model.predict(self.X_pre_train)
            y_test_pred_pls = model.predict(self.X_pre_test)

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

            def load_spectrum(y, y_pred):
                plt.scatter(y, y_pred, label='Data')
                plt.xlabel('Actual Response')
                plt.ylabel('Predicted Response')
                plt.title(f'{name_model_pls} Regression (R²={R_Squared_Test_pls:.2f})')
                reg_pls = np.polyfit(y, y_pred, deg=1)
                trend_pls = np.polyval(reg_pls, y)
                plt.plot(y, trend_pls, 'r', label='Line pred')
                plt.plot(y, y, color='green', linestyle='--', linewidth=1, label="Line fit")
                plt.show()

            load_spectrum(self.y_pre_test, y_test_pred_pls)

        if model_regression == 'Combine':
            name_model_knn = 'Stracking'

            base_models = [
                ('rf', PLSRegression(n_components=7)),
                ('p', KNeighborsRegressor(n_neighbors=90)),
                # ('g', GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=0.1))
            ]

            # model_knn = PLSRegression(n_components=2)
            model_knn = RandomForestRegressor(n_estimators=60, random_state=42, max_depth=5)
            # model_knn = KNeighborsRegressor(n_neighbors=20)
            # model_knn = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=0.1)
            model_knn = StackingRegressor(estimators=base_models, final_estimator=model_knn)

            model_knn.fit(self.X_pre_train, self.y_pre_train)

            y_train_pred_knn = model_knn.predict(self.X_pre_train)
            y_test_pred_knn = model_knn.predict(self.X_pre_test)

            # R, R_Squared, R_MSE
            print('--------------- TRAIN--------------------')
            R_Train_knn = np.corrcoef(self.y_pre_train, y_train_pred_knn, rowvar=False)
            print('R:', "{:.3f}".format(R_Train_knn[0][1]))
            R_Squared_Train_knn = r2_score(self.y_pre_train, y_train_pred_knn)
            print('R^2:', "{:.3f}".format(R_Squared_Train_knn))
            print(f"Accuracy: {R_Squared_Train_knn * 100:.3f}%")
            RMSE_Train_knn = math.sqrt(mean_squared_error(self.y_pre_train, y_train_pred_knn))
            print('R_MSE :', "{:.3f}".format(RMSE_Train_knn))
            MAE_Train_knn = mean_absolute_error(self.y_pre_train, y_train_pred_knn)
            print('MAE:', "{:.3f}".format(MAE_Train_knn))
            # R, R_Squared, R_MSE
            print('--------------- TEST--------------------')
            R_Test_knn = np.corrcoef(self.y_pre_test, y_test_pred_knn, rowvar=False)
            print('R:', "{:.3f}".format(R_Test_knn[0][1]))
            R_Squared_Test_knn = r2_score(self.y_pre_test, y_test_pred_knn)
            print('R^2:', "{:.3f}".format(R_Squared_Test_knn))
            print(f"Accuracy: {R_Squared_Test_knn * 100:.3f}%")
            RMSE_Test_knn = math.sqrt(mean_squared_error(self.y_pre_test, y_test_pred_knn))
            print('R_MSE :', "{:.3f}".format(RMSE_Test_knn))
            MAE_Test_knn = mean_absolute_error(self.y_pre_test, y_test_pred_knn)
            print('MAE:', "{:.3f}".format(MAE_Test_knn))
            print('--------------- RPD--------------------')
            RPD_Test_knn = np.std(self.y_pre_test) / RMSE_Test_knn
            print('RPD:', "{:.2f}".format(RPD_Test_knn))

            def load_spectrum(y, y_pred):
                plt.scatter(y, y_pred, label='Data')
                plt.xlabel('Actual Response')
                plt.ylabel('Predicted Response')
                plt.title(f'{name_model_knn} Regression (R²={R_Squared_Test_knn:.2f})')
                reg_knn = np.polyfit(y, y_pred, deg=1)
                trend_knn = np.polyval(reg_knn, y)
                plt.plot(y, trend_knn, 'r', label='Line pred')
                plt.plot(y, y, color='green', linestyle='--', linewidth=1, label='Line fit')
                plt.show()

            load_spectrum(self.y_pre_test, y_test_pred_knn)


class data_spectrum:

    def __init__(self, path_file_data, name_file, data_split, name_split_column,
                 name_column, non_prepro, path_spectrum, list_data, name_value_split, save_or_none):
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
        df = pd.read_csv(path_file_data + fr'\{name_file}.csv', sep=',')
        df_loc = df[df[f'{name_split_column}'] == name_value_split]

        if data_split == 'Split':
            wavelengths_prepro_spectrum = df_loc.iloc[:0, 12:]
            full_waves_spectrum = [f'{e}' for e in wavelengths_prepro_spectrum]
            for check in range(0, len(list_data)):
                try:
                    list_data_value0 = df_loc[df_loc[f'{name_column}'] == list_data[check]]
                    list_data_value0 = list_data_value0.iloc[:, 12:]
                    prepro_list_data_value0 = msc(list_data_value0)
                    list_data_value0 = pd.DataFrame(list_data_value0, columns=full_waves_spectrum)
                    prepro_list_data_value0 = pd.DataFrame(prepro_list_data_value0, columns=full_waves_spectrum)
                    list_data_value0.to_csv(path_spectrum + r'\non_prepro' + fr'\{list_data[check]}' + '.csv',
                                            index=False,
                                            header=True)
                    prepro_list_data_value0.to_csv(path_spectrum + r'\prepro' + fr'\{list_data[check]}' + '.csv',
                                                   index=False,
                                                   header=True)
                except:
                    pass

            print(f'Successful export data non_prepro_spectrum excel to {path_spectrum}')

            # input column starting waves
            start_col = 1
            if non_prepro == 'None':

                fig, Ax = plt.subplots(2, figsize=(12, 7))
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
                            Data_plot = pd.read_csv(path_spectrum + r'\non_prepro' + fr'\{list_data[check]}.csv',
                                                    sep=',')
                            # load plot data
                            plot(Ax[check], Data_plot, start=start_col, title=f'{list_data[check]}')
                            # load plot mean data
                            plot_mean(Data_plot, start=start_col, label=f'{list_data[check]}')
                    except:
                        print('Not have data', {list_data[check]})

                plt.grid(1)
                plt.legend()
                plt.show()
                if save_or_none == 'Save':
                    plt.savefig(path_spectrum + r'\spectrum' + r'\non_prepro_spectrum.png')
                    plt.savefig(path_spectrum + r'\spectrum' + r'\non_prepro_mean_spectrum.png')
                if save_or_none == 'None':
                    pass

            if non_prepro == 'Prepro':

                fig, Ax = plt.subplots(2, figsize=(12, 7))
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
                            Data_plot = pd.read_csv(path_spectrum + r'\prepro' + fr'\{list_data[check]}.csv',
                                                    sep=',')
                            # load plot data
                            plot(Ax[check], Data_plot, start=start_col, title=f'{list_data[check]}')
                            # load plot mean data
                            plot_mean(Data_plot, start=start_col, label=f'{list_data[check]}')
                    except:
                        print('Not have data', {list_data[check]})

                plt.legend()
                plt.grid(1)
                plt.show()
                if save_or_none == 'Save':
                    plt.savefig(path_spectrum + r'\spectrum' + r'\non_prepro_mean_spectrum.png')
                    plt.savefig(path_spectrum + r'\spectrum' + r'\prepro_spectrum.png')
                if save_or_none == 'None':
                    pass

        if data_split == 'None':
            wavelengths_prepro_spectrum = df.iloc[:0, 12:]
            full_waves_spectrum = [f'{e}' for e in wavelengths_prepro_spectrum]
            for check in range(0, len(list_data)):
                try:
                    list_data_value0 = df[df[f'{name_column}'] == list_data[check]]
                    list_data_value0 = list_data_value0.iloc[:, 12:]
                    prepro_list_data_value0 = msc(list_data_value0)
                    list_data_value0 = pd.DataFrame(list_data_value0, columns=full_waves_spectrum)
                    prepro_list_data_value0 = pd.DataFrame(prepro_list_data_value0, columns=full_waves_spectrum)
                    list_data_value0.to_csv(path_spectrum + r'\non_prepro' + fr'\{list_data[check]}' + '.csv',
                                            index=False,
                                            header=True)
                    prepro_list_data_value0.to_csv(path_spectrum + r'\prepro' + fr'\{list_data[check]}' + '.csv',
                                                   index=False,
                                                   header=True)
                except:
                    pass

            print(f'Successful export data non_prepro_spectrum excel to {path_spectrum}')

            # input column starting waves
            start_col = 1
            if non_prepro == 'None':

                fig, Ax = plt.subplots(2, figsize=(12, 7))
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
                            Data_plot = pd.read_csv(path_spectrum + r'\non_prepro' + fr'\{list_data[check]}.csv',
                                                    sep=',')
                            # load plot data
                            plot(Ax[check], Data_plot, start=start_col, title=f'{list_data[check]}')
                            # load plot mean data
                            plot_mean(Data_plot, start=start_col, label=f'{list_data[check]}')
                    except:
                        print('Not have data', {list_data[check]})

                plt.grid(1)
                plt.legend()
                plt.show()
                if save_or_none == 'Save':
                    plt.savefig(path_spectrum + r'\spectrum' + r'\non_prepro_spectrum.png')
                    plt.savefig(path_spectrum + r'\spectrum' + r'\non_prepro_mean_spectrum.png')
                if save_or_none == 'None':
                    pass

            if non_prepro == 'Prepro':

                fig, Ax = plt.subplots(2, figsize=(12, 7))
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
                            Data_plot = pd.read_csv(path_spectrum + r'\prepro' + fr'\{list_data[check]}.csv',
                                                    sep=',')
                            # load plot data
                            plot(Ax[check], Data_plot, start=start_col, title=f'{list_data[check]}')
                            # load plot mean data
                            plot_mean(Data_plot, start=start_col, label=f'{list_data[check]}')
                    except:
                        print('Not have data', {list_data[check]})

                plt.legend()
                plt.grid(1)
                plt.show()
                if save_or_none == 'Save':
                    plt.savefig(path_spectrum + r'\spectrum' + r'\non_prepro_mean_spectrum.png')
                    plt.savefig(path_spectrum + r'\spectrum' + r'\prepro_spectrum.png')
                if save_or_none == 'None':
                    pass


# chay chuong trinh
if __name__ == "__main__":
    # path data excel
    path_file_data_all = r'D:\Luan Van\Data\Final_Data\Final_Data.csv'
    path_folder_file_data_all = r'D:\Luan Van\Data\Final_Data'

    # path save
    path_save_train_test = r'D:\Luan Van\Data\train_test'

    # --------------------------------------EXPORT DATA-----------------------------------------------------------------
    # path_calib = r'D:\Luan Van\Data\Calib\final_data_calibration.csv'
    # folder_sensor = r'D:\Luan Van\data_sensor\2023-09-30'
    # path_folder_save = r'D:\Luan Van\Data\Demo_Data'
    #
    # data_excel(file_name='Demo_Data_300923_full-waves',
    #            path_folder_sensor=folder_sensor,
    #            path_save=path_folder_save,
    #            path_calib_file=path_calib, list_column=['Ratio', 'Acid', 'Brix', 'Date',
    #                                                     'Point', 'Position', 'Number'],
    #            Cultivar_name='QD')

    # --------------------------------------FIND BEST WAVES-------------------------------------------------------------
    # path_save_file_loc_waves = r'D:\Luan Van\Data\loc_waves'
    #
    # data_find_best_waves(n_estimators=100,
    #                      k_train=100,
    #                      namefile='change_val',
    #                      path_file_data=path_file_data_all,
    #                      path_save=path_save_file_loc_waves)

    # --------------------------------------TRAIN TEST SPLIT------------------------------------------------------------
    path_file_loc_waves = r'D:\Luan Van\Data\loc_waves\change_val_file_waves.csv'

    Preprocessing_data(path_file_data=path_file_data_all,
                       path_file_new_waves=path_file_loc_waves,
                       full_or_part='full',
                       path_save=path_save_train_test)

    # --------------------------------------PREPROCESS DATA-------------------------------------------------------------
    split_data(path_train_test=path_save_train_test, test_size=0.3, prepro_or_none='Prepro')

    # --------------------------------------REGRESSION DATA-------------------------------------------------------------
    data_predict_regression(model_regression='PLS',
                            path_file_data=path_save_train_test)

    # --------------------------------------SPECTRUM PLOT---------------------------------------------------------------
    # path_save_prepro_spectrum = r'D:\Luan Van\Data\spectrum'
    #
    # data_spectrum(name_file='Final_Data', path_file_data=path_folder_file_data_all,
    #               data_split='None', name_split_column='Days late', name_value_split='1 day',
    #               path_spectrum=path_save_prepro_spectrum, save_or_none='None',
    #               name_column='Position', list_data=['Mid of Segments', 'Mid of 2 Segments'],
    #               non_prepro='None')
