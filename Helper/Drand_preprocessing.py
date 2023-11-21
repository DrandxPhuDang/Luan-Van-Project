import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


def preprocessing_data(X_data, windows_len=21, polyorder=2, deriv=1):
    """
    :param windows_len: Nhập vào size cửa sổ trượt
    :param deriv: Nhập vào thứ tự đạo hàm (default = 1)
    :param polyorder: Nhập vào bậc đa thức (default = 2)
    :param X_data: Nhập X_data cần tiền xử lí, nên dùng trước khi chia train_test
    :return: Trả về X_data_preprocessing
    example: X = Preprocessing_data(X)
    """
    X_data = pd.DataFrame(X_data).dropna()
    X_data.fillna(X_data.mean(), inplace=True)
    X_data = preprocessing.normalize(X_data)
    prepro_normal_train = preprocessing.Normalizer().fit(X_data)
    X_data = prepro_normal_train.transform(X_data)
    X_data = savgol_filter(X_data, windows_len, polyorder=polyorder, deriv=deriv)
    scaler_mm = MinMaxScaler()
    X_data = scaler_mm.fit_transform(X_data)
    X_data = pd.DataFrame(np.array(X_data))
    return X_data


def msc_data(input_data):
    """
    :param input_data: X_data: Nhập X_data cần tiền xử lí, nên dùng trước khi chia train_test và tiền xử lí
    :return: Trả về giá trị đã msc
    example: X = msc_data(X)
    """
    _ = np.finfo(np.float32).eps
    input_data = np.array(input_data, dtype=np.float64)
    ref = []
    sampleCount = int(len(input_data))
    for i in range(input_data.shape[0]):
        input_data[i, :] -= input_data[i, :].mean()
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        progress = (i / input_data.shape[0]) * 100
        print(f"\rProgress: %.2f%% ({i}/{input_data.shape[0]})" % progress, end="")
        for j in range(0, sampleCount, 10):
            ref.append(np.mean(input_data[j:j + 10], axis=0))
            fit = np.polyfit(ref[i], input_data[i, :], 1, full=True)
            data_msc[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]
    print("\n-----------------DONE-------------------")
    return data_msc


def remove_outliers_iqr(data, threshold=2, start_col=9, list_col_drop=None):
    """
    :param data: Nhập vào data muốn loại outliers
    :param threshold: Giá trị phân ngưỡng
    :param start_col: Cột bắt đầu lấy data
    :param list_col_drop: List cột muốn bỏ (cột bắt đầu là cột 0)
    :return: Trả về data đã loại outliers
    example: df = remove_outliers_iqr(df, threshold=2, start_col=9, list_col_drop=[(1, 2)])
    """
    data = data.iloc[:, start_col:].copy()  # Use .copy() to create a copy of the DataFrame

    if list_col_drop is not None:
        for col_drop in list_col_drop:
            start_col = col_drop[0]
            end_col = col_drop[1]
            cols_to_drop = data.columns[start_col:end_col + 1]
            data.drop(cols_to_drop, axis=1, inplace=True)

    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    upper_threshold = q3 + threshold * iqr
    lower_threshold = q1 - threshold * iqr
    outliers = (data < lower_threshold) | (data > upper_threshold)
    data_clean = data[~np.any(outliers, axis=1)]

    return data_clean


def remove_outliers_model(X, y, threshold=2, epsilon=0.25, cache_size=50, C=5, kernel='rbf'):
    model = SVR(epsilon=epsilon, cache_size=cache_size, C=C, kernel=kernel)
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred.flatten()
    z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))
    X_clean = X[z_scores <= threshold]
    y_clean = y[z_scores <= threshold]
    return X_clean, y_clean
