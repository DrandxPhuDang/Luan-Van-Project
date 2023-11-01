import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def preprocessing_data(X_data):
    """
    :param X_data: Nhập X_data cần tiền xử lí, nên dùng trước khi chia train_test
    :return: Trả về X_data_preprocessing
    """
    X_data = pd.DataFrame(X_data).dropna()
    X_data.fillna(X_data.mean(), inplace=True)
    X_data = preprocessing.normalize(X_data)
    prepro_normal_train = preprocessing.Normalizer().fit(X_data)
    X_data = prepro_normal_train.transform(X_data)
    X_data = savgol_filter(X_data, 21, polyorder=2, deriv=1)
    scaler_nor = StandardScaler()
    X_data = scaler_nor.fit_transform(X_data)
    X_data = pd.DataFrame(X_data)
    return X_data


def msc_data(input_data):
    """
    :param input_data: X_data: Nhập X_data cần tiền xử lí, nên dùng trước khi chia train_test và tiền xử lí
    :return: Trả về giá trị đã msc
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


def remove_outliers_iqr(data, threshold, start_col, list_col_drop=None):
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
    """
    list_features = df.iloc[:0, start_col:]
    features = [f'{e}' for e in list_features]
    df_num = df[df['Number'] == 1]
    df_num = df_num[df_num['Point'] == 1]
    Features_col_all = pd.DataFrame()
    for p in df_num['Position']:
        df_A = df[df['Position'] == p]
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
            df_N = df_N[df_N['Position'] == 'A']
            mean = 0
            list_mean = []
            for i in features:
                for j in df_N[i].values:
                    mean = j + mean
                mean = mean / 3
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
