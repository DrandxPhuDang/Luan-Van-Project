import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap


def k_pca_reduce(X, y, features):
    """
    :param X: Nhập nên là X_train để giảm chiều (nên dùng fit_transform cho train_data và transform cho test_data)
    :param y: Nhập nên là y_train để giảm chiều (không cần fit_transform hay transform cho y_train)
    :param features: Nhập tất cả cột của X_data
    :return: trả về model tối ưu
    example: self.X_train, self.X_val, self.X_test = reduce_kernel_pca(self.X_train, self.X_val, self.X_test,
                                                            self.y_train, features_col=features, kernel_pca=kernel_pca)
    các tham số trong param_grid không cần thay đổi nếu không cần thiết
    """
    print('Running Kernel PCA Reduce')
    pipeline = Pipeline([
        ('k_pca', KernelPCA()),
        ('svm', SVR())
    ])
    param_grid = {
        'k_pca__n_components': np.arange(1, len(features), 1),
        # 'k_pca__n_components': [5],
        'k_pca__kernel': ['linear', 'rbf', 'sigmoid']
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=10, verbose=0)
    grid_search.fit(X, y)
    k_pca = KernelPCA(n_components=grid_search.best_params_["k_pca__n_components"],
                      kernel=grid_search.best_params_["k_pca__kernel"])
    print('Dimensions: ', grid_search.best_params_["k_pca__n_components"])
    print('Kernel: ', grid_search.best_params_["k_pca__kernel"])
    return k_pca


def isomap_reduce():
    isomap = Isomap(n_components=100)
    return isomap
