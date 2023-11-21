import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

"""
    Tổng hợp các model sử dụng grid_search, các parameter đã tham khảo tài liệu và lựa chọn tối ưu
    __Write by Drand__
    example:    best_model_r = Gridsearch_r(self.X_val, self.y_val)
                self.model = best_model_r.best_estimator_
                self.model.fit(self.X_train, self.y_train)
                print('Best_params_: ', best_model_r.best_params_)
                
"""

k_fold = KFold(n_splits=10, shuffle=True, random_state=42)


def Gridsearch_ann(X, y):
    """
    :param X: Nhập X để tìm tham số tối ưu thường là X_train hoặc X_val
    :param y: Nhập y để tìm tham số tối ưu thường là y_train hoặc y_val
    :return: Trả về model (chọn tham số tối ưu, hoặc model chứa tham số tối ưu)
    """
    param_grid_ann = {
        'hidden_layer_sizes': [(16, 16), (32, 16), (64, 16), (16, 32), (32, 32), (64, 32),
                               (16, 64), (32, 64), (64, 64)],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1]
    }
    ann = MLPRegressor(random_state=42, early_stopping=False)
    grid_search_ann = GridSearchCV(estimator=ann, param_grid=param_grid_ann, cv=k_fold, verbose=0, n_jobs=-1,
                                   scoring='neg_mean_squared_error')

    grid_search_ann.fit(X, y)
    return grid_search_ann


def Gridsearch_rf(X, y):
    """
    :param X: Nhập X để tìm tham số tối ưu thường là X_train hoặc X_val
    :param y: Nhập y để tìm tham số tối ưu thường là y_train hoặc y_val
    :return: Trả về model (chọn tham số tối ưu, hoặc model chứa tham số tối ưu)
    """
    param_grid_rf = {
        'n_estimators': np.arange(10, 200, 10),
        'max_depth': np.arange(2, 10, 2),
    }
    rf = RandomForestRegressor(random_state=42, bootstrap=True)
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=k_fold, verbose=0, n_jobs=-1,
                                  scoring='neg_mean_squared_error')
    grid_search_rf.fit(X, y)
    return grid_search_rf


def Gridsearch_svr(X, y):
    """
    :param X: Nhập X để tìm tham số tối ưu thường là X_train hoặc X_val
    :param y: Nhập y để tìm tham số tối ưu thường là y_train hoặc y_val
    :return: Trả về model (chọn tham số tối ưu, hoặc model chứa tham số tối ưu)
    """
    param_grid_svr = {
        'C': np.arange(5, 300, 5),
        'epsilon': np.arange(0.05, 1, 0.05),
        'cache_size': np.arange(50, 200, 50),
    }
    svr = SVR(kernel='rbf', gamma='scale', shrinking=True)
    grid_search_svr = GridSearchCV(estimator=svr, param_grid=param_grid_svr, cv=k_fold, verbose=0, n_jobs=-1,
                                   scoring='neg_mean_squared_error')
    grid_search_svr.fit(X, y)
    return grid_search_svr


def Gridsearch_r(X, y):
    """
    :param X: Nhập X để tìm tham số tối ưu thường là X_train hoặc X_val
    :param y: Nhập y để tìm tham số tối ưu thường là y_train hoặc y_val
    :return: Trả về model (chọn tham số tối ưu, hoặc model chứa tham số tối ưu)
    """
    param_grid_r = {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    }
    rd = Ridge()
    grid_search_r = GridSearchCV(estimator=rd, param_grid=param_grid_r, cv=k_fold, verbose=0, n_jobs=-1,
                                 scoring='neg_mean_squared_error')
    grid_search_r.fit(X, y)
    return grid_search_r


def Gridsearch_l(X, y):
    """
    :param X: Nhập X để tìm tham số tối ưu thường là X_train hoặc X_val
    :param y: Nhập y để tìm tham số tối ưu thường là y_train hoặc y_val
    :return: Trả về model (chọn tham số tối ưu, hoặc model chứa tham số tối ưu)
    """
    param_grid_l = {
        'alpha': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    }
    ls = Lasso(tol=1e-2)
    grid_search_l = GridSearchCV(estimator=ls, param_grid=param_grid_l, cv=k_fold, verbose=0, n_jobs=-1,
                                 scoring='neg_mean_squared_error')
    grid_search_l.fit(X, y)
    return grid_search_l


def Gridsearch_knn(X, y):
    """
    :param X: Nhập X để tìm tham số tối ưu thường là X_train hoặc X_val
    :param y: Nhập y để tìm tham số tối ưu thường là y_train hoặc y_val
    :return: Trả về model (chọn tham số tối ưu, hoặc model chứa tham số tối ưu)
    """
    param_grid_knn = {
        'n_neighbors': np.arange(5, 100, 5),
        'leaf_size': np.arange(2, 50, 2)
    }
    knn = KNeighborsRegressor(algorithm='auto')
    grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=k_fold, verbose=0, n_jobs=-1,
                                   scoring='neg_mean_squared_error')
    grid_search_knn.fit(X, y)
    return grid_search_knn


def Gridsearch_dt(X, y):
    """
    :param X: Nhập X để tìm tham số tối ưu thường là X_train hoặc X_val
    :param y: Nhập y để tìm tham số tối ưu thường là y_train hoặc y_val
    :return: Trả về model (chọn tham số tối ưu, hoặc model chứa tham số tối ưu)
    """
    param_grid_dt = {
        'max_depth': np.arange(2, 10, 2),
    }
    dt = DecisionTreeRegressor()
    grid_search_dt = GridSearchCV(estimator=dt, param_grid=param_grid_dt, cv=k_fold, verbose=0, n_jobs=-1,
                                  scoring='neg_mean_squared_error')
    grid_search_dt.fit(X, y)
    return grid_search_dt


def Gridsearch_pls(X, y, features):
    """
    :param X: Nhập X để tìm tham số tối ưu thường là X_train hoặc X_val
    :param y: Nhập y để tìm tham số tối ưu thường là y_train hoặc y_val
    :param features: là độ dài của bước sóng trước khi chọn ra đặt trưng (feature X của data)
    Chỉ có PLS mới cần thêm tham số features vì PLS chọn ra các đặt trưng từ X_data
    :return: Trả về model (chọn tham số tối ưu, hoặc model chứa tham số tối ưu)
    """
    list_len_features = []
    for value in range(1, len(features)+1):
        list_len_features.append(value)
    param_grid_pls = {
        'n_components': list_len_features
    }
    pls = PLSRegression()
    grid_search_pls = GridSearchCV(estimator=pls, param_grid=param_grid_pls, cv=k_fold, verbose=0, n_jobs=-1,
                                   scoring='neg_mean_squared_error')
    grid_search_pls.fit(X, y)
    return grid_search_pls


def Gridsearch_xgb(X, y):
    """
    :param X: Nhập X để tìm tham số tối ưu thường là X_train hoặc X_val
    :param y: Nhập y để tìm tham số tối ưu thường là y_train hoặc y_val
    :return: Trả về model (chọn tham số tối ưu, hoặc model chứa tham số tối ưu)
    """
    param_grid_XGB = {
        'max_depth': np.arange(2, 10, 2),
        'n_estimators': np.arange(10, 200, 10),
    }
    XGB = xgb.XGBRegressor()
    grid_search_XGB = GridSearchCV(estimator=XGB, param_grid=param_grid_XGB, cv=k_fold, verbose=0, n_jobs=-1,
                                   scoring='neg_mean_squared_error')
    grid_search_XGB.fit(X, y)
    return grid_search_XGB


def Gridsearch_gbr(X, y):
    """
    :param X: Nhập X để tìm tham số tối ưu thường là X_train hoặc X_val
    :param y: Nhập y để tìm tham số tối ưu thường là y_train hoặc y_val
    :return: Trả về model (chọn tham số tối ưu, hoặc model chứa tham số tối ưu)
    """
    param_grid_gbr = {
        'n_estimators': np.arange(10, 200, 10),
        'max_depth': np.arange(2, 10, 2)
    }
    gbr = GradientBoostingRegressor(random_state=42)
    grid_search_gbr = GridSearchCV(estimator=gbr, param_grid=param_grid_gbr, cv=k_fold, verbose=0, n_jobs=-1,
                                   scoring='neg_mean_squared_error')
    grid_search_gbr.fit(X, y)
    return grid_search_gbr


def Gridsearch_lr(X, y):
    """
    :param X: Input X để tìm tham số tối ưu thường là X_train hoặc X_val
    :param y: Nhập y để tìm tham số tối ưu thường là y_train hoặc y_val
    :return: Trả về model (chọn tham số tối ưu, hoặc model chứa tham số tối ưu)
    """
    param_grid_lr = {
        'fit_intercept': [True, False]
    }
    lr = LinearRegression()
    grid_search_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=k_fold, verbose=0, n_jobs=-1,
                                  scoring='neg_mean_squared_error')
    grid_search_lr.fit(X, y)
    return grid_search_lr


def Gridsearch_etr(X, y):
    """
    :param X: Nhập X để tìm tham số tối ưu thường là X_train hoặc X_val
    :param y: Nhập y để tìm tham số tối ưu thường là y_train hoặc y_val
    :return: Trả về model (chọn tham số tối ưu, hoặc model chứa tham số tối ưu)
    """
    param_grid_etr = {
        'n_estimators': np.arange(10, 200, 10),
        'max_depth': np.arange(2, 10, 2)
    }
    etr = ExtraTreesRegressor(random_state=42)
    grid_search_etr = GridSearchCV(estimator=etr, param_grid=param_grid_etr, cv=k_fold, verbose=0, n_jobs=-1,
                                   scoring='neg_mean_squared_error')
    grid_search_etr.fit(X, y)
    return grid_search_etr


def Gridsearch_adb(X, y):
    """
    :param X: Nhập X để tìm tham số tối ưu thường là X_train hoặc X_val
    :param y: Nhập y để tìm tham số tối ưu thường là y_train hoặc y_val
    :return: Trả về model (chọn tham số tối ưu, hoặc model chứa tham số tối ưu)
    """
    param_grid_adb = {
        'n_estimators': np.arange(10, 200, 10),
    }
    adb = AdaBoostRegressor(random_state=42)
    grid_search_adb = GridSearchCV(estimator=adb, param_grid=param_grid_adb, cv=k_fold, verbose=0, n_jobs=-1,
                                   scoring='neg_mean_squared_error')
    grid_search_adb.fit(X, y)
    return grid_search_adb
