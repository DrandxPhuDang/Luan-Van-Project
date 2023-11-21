import math
import numpy as np
import pandas as pd
from kennard_stone import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from Helper.Drand_k_pca import k_pca_reduce
from Helper.Drand_preprocessing import preprocessing_data


def get_data_X_y(df_all, start_col, mean_features_data=False, pick_features_data=False):
    """
    :param pick_features_data:
    :param mean_features_data: Khi nhập vào data ngẫu nhiên mới cần mean_features và pick_one_features
    :param df_all: Nhập vào data file csv
    :param start_col: Nhập vào số thứ tự của cột bắt đầu bước sóng
    :return: Trả về 3 giá trị, X là giá trị dãy bước sóng, y là giá trị brix, features là giá trị bước sóng
    example: df = pd.read_csv(path_file_data)
             X, y, features = get_data_X_y(df, start_col=start_col_X)
    """
    X_all = pd.DataFrame()
    y_all = pd.DataFrame()
    features_all = pd.DataFrame()
    if (mean_features_data is False) & (pick_features_data is False):
        list_features = df_all.iloc[:0, start_col:]
        features_all = [f'{e}' for e in list_features]
        X_all = df_all[features_all]
        y_all = df_all['Brix']
    if mean_features_data is True:
        df_mean = mean_features(df_all, path_save_file=r'D:\Luan Van\Data\Final_Data\Random_mean_measuring.csv',
                                start_col=start_col)
        list_features = df_mean.iloc[:0, 4:]
        features_all = [f'{e}' for e in list_features]
        X_all = df_mean[features_all]
        y_all = df_mean['Brix']
    else:
        pass

    if pick_features_data is True:
        pick_features(df_all, path_save_file=r'D:\Luan Van\Data\Final_Data\Random_pick_measuring.csv')
        df_mean = pd.read_csv(r'D:\Luan Van\Data\Final_Data\Random_pick_measuring.csv')
        list_features = df_mean.iloc[:0, start_col:]
        features_all = [f'{e}' for e in list_features]
        X_all = df_mean[features_all]
        y_all = df_mean['Brix']
    else:
        pass

    return X_all, y_all, features_all


def train_test_split_kennard_stone(X_data, y_data, test_size, prepro_data, features):
    """
    :param features:
    :param X_data: Nhập vào X data
    :param y_data: Nhập vào y data
    :param test_size: Chọn test_size
    :param prepro_data: Dùng để chọn dữ liệu có tiền xử lí hay không
    :return: X_train, X_val, X_test, y_train, y_val, y_test, bộ dữ liệu được chia theo test_size, tệp val cố định bằng
    20% tệp train
    example; self.X_train, self.X_val, self.X_test, \
            self.y_train, self.y_val, self.y_test = train_test_split_kennard_stone(X, y, test_size, prepro_data)
    """
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    X_val = pd.DataFrame()
    y_train = pd.DataFrame()
    y_val = pd.DataFrame()
    y_test = pd.DataFrame()
    if prepro_data is True:
        X_data = preprocessing_data(X_data)
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    if prepro_data is False:
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    def save_train_test():
        all_data = pd.concat([pd.DataFrame(np.array(y_data), columns=['Brix']),
                              pd.DataFrame(np.array(X_data), columns=features)], axis=1)
        train_all = pd.concat([pd.DataFrame(np.array(y_train), columns=['Brix']),
                               pd.DataFrame(np.array(X_train), columns=features)], axis=1)
        test_all = pd.concat([pd.DataFrame(np.array(y_test), columns=['Brix']),
                              pd.DataFrame(np.array(X_test), columns=features)], axis=1)
        val_all = pd.concat([pd.DataFrame(np.array(y_val), columns=['Brix']),
                             pd.DataFrame(np.array(X_val), columns=features)], axis=1)
        all_data.to_csv(r'D:\Luan Van\Data\train_test\all.csv', index=False, header=True, na_rep='Unknown')
        train_all.to_csv(r'D:\Luan Van\Data\train_test\train.csv', index=False, header=True, na_rep='Unknown')
        test_all.to_csv(r'D:\Luan Van\Data\train_test\test.csv', index=False, header=True, na_rep='Unknown')
        val_all.to_csv(r'D:\Luan Van\Data\train_test\val.csv', index=False, header=True, na_rep='Unknown')
    save_train_test()

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
    actual_values = np.ravel(actual_values)
    predictions = np.ravel(predictions)
    sd_actual = np.std(actual_values)
    error = (predictions - actual_values)
    bias = np.mean(predictions - actual_values)
    sep = np.sqrt(np.mean((error - bias) ** 2))
    rpd = sd_actual / sep
    return rpd


def plot_spectrum(y_actual, y_pred, name_model, score_test):
    """
    :param y_actual: Nhập y_data từ dãy X_data đã dự đoán
    :param y_pred: Nhập y_pred
    :param name_model: Tên cho biểu đồ
    :param score_test: score_test là lấy giá trị R_Square của test để tính độ tin cậy và xuất trên biểu đồ
    :return: Không trả về biến nào
    example: '''Plot Regression'''
            load_spectrum(self.y_test, y_pred_test, name_model=self.name_model, score_test=score_test)
    """
    plt.figure(num=1)
    plt.scatter(y_actual, y_pred, label='Data')
    plt.xlabel('Actual Response')
    plt.ylabel('Predicted Response')
    plt.title(f'{name_model} Regression (R²={score_test[1]:.2f})')
    reg_pls = np.polyfit(y_actual, y_pred, deg=1)
    trend_pls = np.polyval(reg_pls, y_actual)
    plt.plot(y_actual, trend_pls, 'r', label='Line pred')
    plt.plot(y_actual, y_actual, color='green', linestyle='--', linewidth=1, label="Line fit")


def plot_brix(data, bins=20):
    """
    :param data: Dữ liệu cần vẽ sự phân bố
    :param bins: Độ rộng của trục số lượng dữ liệu
    :return: Hiển thị biểu đồ histogram
    """
    plt.figure(num=2)
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title('Brix Distribution')
    plt.xlabel('Brix')
    plt.ylabel('Frequency')


def mean_features(df, path_save_file, start_col):
    """
    :param df: file data csv đo ngẫu nhiên
    :param path_save_file: địa chỉ lưu file
    :param start_col: số cột bắt đầu
    :return: data frame đã tính trung bình phổ 3 điểm
    """

    # ----------------- get number -----------------------
    def get_number(data):
        data_area = data[data['Area'] == 'A']
        data_point = data_area[data_area['Point'] == 1]
        data_number = data_point['Number']
        return data_number

    # ----------------- get area -----------------------
    def get_area(number):
        data_number = df[df['Number'] == number]
        data_point = data_number[data_number['Point'] == 1]
        data_area = data_point['Area']
        return data_area

    # ----------------- get brix -----------------------
    def get_brix(data):
        data_point = data[data['Point'] == 1]
        data_brix = data_point['Brix']
        return data_brix

    def get_acid(data):
        data_point = data[data['Point'] == 1]
        data_acid = data_point['Acid']
        return data_acid

    def get_ratio(data):
        data_point = data[data['Point'] == 1]
        data_ratio = data_point['Ratio']
        return data_ratio

    list_features = df.iloc[:0, start_col:]
    features = [f'{e}' for e in list_features]
    Features_col_all = pd.DataFrame()
    List_number = []
    for num in get_number(df):
        list_all = []
        for area in get_area(num):
            List_number.append(num)
            df_dt = df[df['Number'] == num]
            df_dt = df_dt[df_dt['Area'] == area]
            features_col = df_dt[features]
            list_mean = []
            mean = 0
            for i in features:
                for j in features_col[i].values:
                    mean = j + mean
                mean = mean / len(features_col[i].values)
                list_mean.append(mean)
                mean = 0
            list_all.append(list_mean)
        Features_col = pd.DataFrame(np.array(list_all), columns=features)
        Features_col_all = pd.DataFrame(Features_col_all)._append([Features_col])

    Features_col_all = Features_col_all.reset_index(drop=True)
    df_brix = get_brix(df).reset_index(drop=True)
    df_acid = get_acid(df).reset_index(drop=True)
    df_ratio = get_ratio(df).reset_index(drop=True)
    df_number = pd.DataFrame(np.array(List_number), columns=['Number']).reset_index(drop=True)

    Features_col_all = pd.concat([df_ratio, Features_col_all], axis=1)
    Features_col_all = pd.concat([df_acid, Features_col_all], axis=1)
    Features_col_all = pd.concat([df_brix, Features_col_all], axis=1)
    Features_col_all = pd.concat([df_number, Features_col_all], axis=1)
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
    data_pick = data[data['Point'] == col_pick]
    data_pick.to_csv(path_save_file, index=False, header=True, na_rep='Unknown')
    return data_pick
