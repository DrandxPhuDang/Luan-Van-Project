import time
import math
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_predict


class DrandRgression:
    def fit(self, X, y):
        self.X = X
        self.y = y
        """
        base_model{
        ExtraTreesRegressor
        XGBoostingRegressor
        SVRegression
        PLSRegreison
        }
        --> model meta: Ridge
        """
        # Creating base model
        base_model = [SVR(),
                      ExtraTreesRegressor(random_state=42),
                      xgb.XGBRegressor(),
                      PLSRegression()
                      ]

        df_y_pred_list = pd.DataFrame()
        cnt = len(base_model)
        for model in base_model:
            cnt = cnt - 1
            model = model
            model.fit(self.X, self.y)
            y_pred = model.predict(X)
            df_y_pred = pd.DataFrame(np.array(y_pred), columns=[cnt]).reset_index(drop=True)
            self.df_y_pred_list = pd.concat([df_y_pred, df_y_pred_list], axis=1)

        self.model = Ridge()
        self.model.fit(self.df_y_pred_list, self.y)

    def predict(self, X=None):
        # predict
        predicted = cross_val_predict(self.model, self.df_y_pred_list, self.y, cv=20)
        return predicted
