import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


def preprocessing_data(X_data):
    """
    :param X_data: Nhập X_data cần tiền xử lí, nên dùng trước khi chia train_test
    :return: Trả về X_data_preprocessing
    """
    X_data = pd.DataFrame(X_data).dropna()
    X_data.fillna(X_data.mean(), inplace=True)
    X_data = preprocessing.normalize(X_data.values)
    prepro_normal_train = preprocessing.Normalizer().fit(X_data)
    X_data = prepro_normal_train.transform(X_data)
    X_data = savgol_filter(X_data, window_length=12, polyorder=1)
    scaler_nor = StandardScaler()
    X_data = scaler_nor.fit_transform(X_data)
    X_data = pd.DataFrame(X_data)
    return X_data


def msc(input_data):
    """
    :param input_data: X_data: Nhập X_data cần tiền xử lí, nên dùng trước khi chia train_test và tiền xử lí
    :return:
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
