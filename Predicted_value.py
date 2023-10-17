import math

import numpy as np
import pandas as pd
from joblib import dump, load
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import scale

model = load(r"D:\Luan Van\model_predicted.pkl")

df = pd.read_csv(r"D:\Luan Van\Data\Final_Data\Final_Data_161023.csv")
list_features = df.iloc[:0, 12:]
features = [f'{e}' for e in list_features]
X = df[features]
y = df['Brix']


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
    X_data = savgol_filter(X_data, window_length=12, polyorder=1)
    # Min Max Scaler
    scaler_X_train = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_data = scaler_X_train.fit_transform(X_data)
    X_data = scaler_X_train.inverse_transform(X_data)
    X_data = pd.DataFrame(X_data)
    return X_data


X = preprocessing_data(X)

# Predicted Values
y_pred_test = model.predict(X)
print("Evaluating Regression Models")


# ------------------------------------------------ Print Score -------------------------------------------------
def print_score(y_actual, y_predicted):
    # R, R_Squared, R_MSE
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


print('--------------- TEST--------------------')
score_test = print_score(y, y_pred_test)
print('--------------- RPD--------------------')
RPD_Test = np.std(y) / score_test[2]
print('RPD:', "{:.2f}".format(RPD_Test))


# ------------------------------------------------ Load Spectrum -----------------------------------------------
def load_spectrum(y_actual, y_pred):
    plt.scatter(y_actual, y_pred, label='Data')
    plt.xlabel('Actual Response')
    plt.ylabel('Predicted Response')
    plt.title(f'Regression (R²={score_test[1]:.2f})')
    reg_pls = np.polyfit(y_actual, y_pred, deg=1)
    trend_pls = np.polyval(reg_pls, y_actual)
    plt.plot(y_actual, trend_pls, 'r', label='Line pred')
    plt.plot(y_actual, y_actual, color='green', linestyle='--', linewidth=1, label="Line fit")
    plt.show()


load_spectrum(y, y_pred_test)
