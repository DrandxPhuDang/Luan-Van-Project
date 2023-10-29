import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


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
