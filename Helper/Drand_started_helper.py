import math
import time

import numpy as np
import pandas as pd
from kennard_stone import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from Helper.Drand_k_pca import k_pca_reduce
from Helper.Drand_preprocessing import preprocessing_data, msc_data, remove_outliers_model


def get_data_X_y(df_all, start_col, mean_features_data=False, pick_features_data=False):
    """
    :param pick_features_data:
    :param mean_features_data: Khi nhập vào data ngẫu nhiên mới cần mean_features và pick_one_features
    :param df_all: Nhập vào data file csv
    :param start_col: Nhập vào số thứ tự của cột bắt đầu bước sóng
    :return: Trả về 3 giá trị, X là giá trị dãy bước sóng, y là giá trị brix, features là sách bước sóng
    example: df = pd.read_csv(path_file_data)
             X, y, features = get_data_X_y(df, start_col=start_col_X)
    """
    list_features = df_all.iloc[:0, start_col:]
    features_all = [f'{e}' for e in list_features]
    X_all = df_all[features_all]
    y_all = df_all['Brix']
    if mean_features_data is True:
        df_mean = mean_features(df_all, path_save_file=r'D:\Luan Van\Data\Final_Data\Random_mean_measuring.csv',
                                start_col=start_col)
        list_features = df_mean.iloc[:0, 2:]
        features_all = [f'{e}' for e in list_features]
        X_all = df_mean[features_all]
        y_all = df_mean['Brix']
    else:
        pass

    if pick_features_data is True:
        df_mean = pick_features(df_all, path_save_file=r'D:\Luan Van\Data\Final_Data\Random_pick_measuring.csv')
        list_features = df_mean.iloc[:0, start_col:]
        features_all = [f'{e}' for e in list_features]
        X_all = df_mean[features_all]
        y_all = df_mean['Brix']
    else:
        pass

    return X_all, y_all, features_all


def train_test_split_kennard_stone(X_data, y_data, test_size, prepro_data):
    """
    :param X_data: Nhập vào X data
    :param y_data: Nhập vào y data
    :param test_size: Chọn test_size
    :param prepro_data: Dùng để chọn dữ liệu có tiền xử lí hay không
    :return: X_train, X_val, X_test, y_train, y_val, y_test, bộ dữ liệu được chia theo test_size, tệp val cố định bằng
    20% tệp train
    example; self.X_train, self.X_val, self.X_test, \
            self.y_train, self.y_val, self.y_test = train_test_split_kennard_stone(X, y, test_size, prepro_data)
    """
    global X_train, X_test, X_val, y_train, y_val, y_test
    if prepro_data is True:
        X_data = preprocessing_data(X_data)
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    if prepro_data is False:
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    return X_train, X_val, X_test, y_train, y_val, y_test


def reduce_kernel_pca(X_fit, X_transform_1, X_transform_2, y_fit, features_col, kernel_pca):
    """
    :param X_fit: Nhập vào X dùng để học dữ liệu giảm chiều (X_train)
    :param X_transform_1: Nhập vòa X cần transform theo dữ liệu tệp X_fit (X_val)
    :param X_transform_2: Nhập vòa X cần transform theo dữ liệu tệp X_fit (X_test)
    :param y_fit: Chạy giảm chiều Kernnel PCA
    :param features_col: Số lượng cột ở tệp X
    :param kernel_pca: Chọn xem có cần giảm chiều không
    :return: Các dữ liệu đã Kernel PCA hoặc không
    example: Reduce Features Data
        self.X_train, self.X_val, self.X_test = reduce_kernel_pca(self.X_train, self.X_val, self.X_test, self.y_train,
                                                                  features_col=features, kernel_pca=kernel_pca)
    """
    if kernel_pca is True:
        k_pca = k_pca_reduce(X_fit, y_fit, features=features_col)
        X_fit = k_pca.fit_transform(X_fit)
        X_transform_1 = k_pca.transform(X_transform_1)
        X_transform_2 = k_pca.transform(X_transform_2)
    if kernel_pca is False:
        pass
    return X_fit, X_transform_1, X_transform_2


def warning(model_regression):
    """
    :param model_regression: Đưa vào tên model để check xem model này đã có trong bộ dữ liệu model không
    :return: In ra các thông báo để kiểm tra
    """
    list_model = ["SVR", "RF", "Stacking", "ANN", "R", "L", "XGB", "PLS", "GBR", "KNN", "DT", "LR", "ETR", "ADB"]
    cnt = 0
    for model in list_model:
        cnt += 1
        if model_regression == model:
            print(f"Notice: Model {model_regression} regression is available")
            print("Creating and training model...")
            break
        else:
            if cnt == len(list_model):
                print(f"Notice: Model {model_regression} regression is not available")
                print(f"Available model: SVR, RF, PLS, ANN, R, XGB, PLS, DT, LR, KNN, GBR and Stacking")


def print_best_params(list_model):
    """
    :param list_model: Dãy model đã được chạy gridsearch
    :return: In ra các parameter tốt nhất theo tiêu chí R Square
    example: '''Print Best parameter'''
                print_best_params([best_model_etr, best_model_r, best_model_xgb,
                                   best_model_pls, best_model_svr])
    """
    for model in list_model:
        try:
            print(model.best_params_)
        except:
            pass


def print_score(y_actual, y_predicted):
    """
    :param y_actual: Nhập y_data từ dãy X_data đã dự đoán
    :param y_predicted: Nhập y_pred
    :return: Trả về 4 đánh giá (R, R Square, R_MSE, MAE)
    example: '''Accuracy score'''
            print('--------------- TRAIN--------------------')
            print_score(self.y_train, y_pred_train)
            print('--------------- TEST--------------------')
            score_test = print_score(self.y_test, y_pred_test)
            Để lấy giá trị R, R_Squared, R_MSE, MAE thì chỉ cần thêm Score_test[0] (0-3)
            tương ứng thứ tự của các giá trị
    """
    R = np.corrcoef(y_actual, y_predicted, rowvar=False)
    print('R:', "{:.3f}".format(R[0][1]))
    R_Squared = r2_score(y_actual, y_predicted)
    print('R^2:', "{:.3f}".format(R_Squared))
    print(f"Accuracy: {R_Squared * 100:.3f}%")
    R_MSE = math.sqrt(mean_squared_error(y_actual, y_predicted))
    print('R_MSE :', "{:.3f}".format(R_MSE))
    MAE = mean_absolute_error(y_actual, y_predicted)
    print('MAE:', "{:.3f}".format(MAE))
    return R, R_Squared, R_MSE, MAE


def cal_rpd(actual_values, predictions):
    """
    :param actual_values: Nhập vào giá trị y thực tế
    :param predictions: Nhập vào giá trị y dự đoán
    :return: Trả về kết quả RPD
    example: print('--------------- RPD--------------------')
            RPD_Test = cal_rpd(self.y_test, y_pred_test)
            print('RPD:', "{:.2f}".format(RPD_Test))
    """
    sd_actual = np.std(actual_values)
    error = (predictions - actual_values)
    bias = np.mean(predictions - actual_values)
    sep = np.sqrt(np.mean((error - bias) ** 2))
    rpd = sd_actual / sep
    return rpd


def load_spectrum(y_actual, y_pred, name_model, score_test):
    """
    :param y_actual: Nhập y_data từ dãy X_data đã dự đoán
    :param y_pred: Nhập y_pred
    :param name_model: Tên cho biểu đồ
    :param score_test: score_test là lấy giá trị R_Square của test để tính độ tin cậy và xuất trên biểu đồ
    :return: Không trả về biến nào
    example: '''Plot Regression'''
            load_spectrum(self.y_test, y_pred_test, name_model=self.name_model, score_test=score_test)
    """
    plt.scatter(y_actual, y_pred, label='Data')
    plt.xlabel('Actual Response')
    plt.ylabel('Predicted Response')
    plt.title(f'{name_model} Regression (R²={score_test[1]:.2f})')
    reg_pls = np.polyfit(y_actual, y_pred, deg=1)
    trend_pls = np.polyval(reg_pls, y_actual)
    plt.plot(y_actual, trend_pls, 'r', label='Line pred')
    plt.plot(y_actual, y_actual, color='green', linestyle='--', linewidth=1, label="Line fit")
    plt.show()


def mean_features(df, path_save_file, start_col):
    """
    :param df: File csv cần tính trung bình phổ
    :param path_save_file: Đường dẫn lưu file
    :param start_col: cột bắt đầu bước sóng
    :return: Trả về dataframe đã tính trung bình
    example:
            def df_mean_features(df_mean, start_col):
            df_mean = mean_features(df_mean, path_save_file=r'D:\Luan Van\Data\Final_Data\mean_features.csv',
                                    start_col=start_col)
            list_features = df_mean.iloc[:0, 2:]
            features_mean = [f'{e}' for e in list_features]
            X_mean = df_mean[features_mean]
            y_mean = df_mean['Brix']
            return X_mean, y_mean, features_mean

    Tính bước sóng trung bình dùng để cho bộ dữ liệu ngẫu nhiên đo 3 điểm, cần lấy dữ liệu phổ trung bình
    """
    list_features = df.iloc[:0, start_col:]
    features = [f'{e}' for e in list_features]
    df_num = df[df['Number'] == 1]
    df_num = df_num[df_num['Point'] == 1]
    Features_col_all = pd.DataFrame()
    for p in df_num['Area']:
        df_A = df[df['Area'] == p]
        df_A1 = df_A[df_A['Point'] == 1]
        list_all = []
        Brix = []
        for b in df_A1['Brix']:
            Brix.append(b)
        Brix_col = pd.DataFrame(np.array(Brix))
        num_list = []
        for a in df_A1['Number']:
            num_list.append(a)
            df_N = df[df['Number'] == a]
            df_N = df_N[df_N['Area'] == 'A']
            mean = 0
            list_mean = []
            for i in features:
                for j in df_N[i].values:
                    mean = j + mean
                mean = mean / len(df_N[i].values)
                list_mean.append(mean)
                mean = 0
            list_all.append(list_mean)
            list_mean = []
        Features_col = pd.DataFrame(np.array(list_all), columns=features)
        Features_col.insert(loc=0, column='Brix', value=Brix_col)
        Features_col.insert(loc=0, column='Number', value=pd.DataFrame(np.array(num_list)))
        Features_col_all = pd.DataFrame(Features_col_all).append([Features_col])
        Features_col = pd.DataFrame()
        Features_col_all.to_csv(path_save_file, index=False, header=True, na_rep='Unknown')
    return Features_col_all


def pick_features(data, path_save_file, col_pick=1):
    """
    :param data: Nhập vào file data cần chọn bước sóng
    :param path_save_file: Nơi lưu file data sau khi chọn
    :param col_pick: Đối tượng được chọn
    :return: Trả về X, y, và dãy bước sóng
    example:
    """
    data = data[data['Point'] == col_pick]
    data.to_csv(path_save_file, index=False, header=True, na_rep='Unknown')
    return data
