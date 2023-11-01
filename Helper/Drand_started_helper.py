import math
import numpy as np
from kennard_stone import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from Helper.Drand_k_pca import k_pca_reduce
from Helper.Drand_preprocessing import preprocessing_data


def get_data_X_y(df_all, start_col):
    """
    :param df_all: Nhập vào data file csv
    :param start_col: Nhập vào số thứ tự của cột bắt đầu bước sóng
    :return: Trả về 3 giá trị, X là giá trị dãy bước sóng, y là giá trị brix, features là sách bước sóng
    """
    list_features = df_all.iloc[:0, start_col:]
    features_all = [f'{e}' for e in list_features]
    X_all = df_all[features_all]
    y_all = df_all['Brix']
    return X_all, y_all, features_all


def train_test_split_kennard_stone(X_data, y_data, test_size, prepro_data):
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
    if kernel_pca is True:
        k_pca = k_pca_reduce(X_fit, y_fit, features=features_col)
        X_fit = k_pca.fit_transform(X_fit)
        X_transform_1 = k_pca.transform(X_transform_1)
        X_transform_2 = k_pca.transform(X_transform_2)
    if kernel_pca is False:
        pass
    return X_fit, X_transform_1, X_transform_2


def warning(model_regression):
    list_model = ["SVR", "RF", "Stacking", "ANN", "R", "L", "XGB", "PLS", "GBR", "KNN", "DT", "LR"]
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
    for model in list_model:
        print(model.best_params_)


def print_score(y_actual, y_predicted):
    """
    :param y_actual: Nhập y_data từ dãy X_data đã dự đoán
    :param y_predicted: Nhập y_pred
    :return: Trả về 4 đánh giá (R, R Square, R_MSE, MAE)
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
