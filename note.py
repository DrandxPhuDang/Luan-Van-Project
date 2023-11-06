import joblib
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import StackingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

from Helper.Drand_started_helper import get_data_X_y, print_score

data_ = r"D:\Luan Van\Data\Final_Data\Random_mean_measuring.csv"
loaded_model = joblib.load(r'D:\Luan Van\model\Support Vector_regression.pkl')
df = pd.read_csv(data_)

list_features = df.iloc[:0, 2:]
features_all = [f'{e}' for e in list_features]
X_all = df[features_all]
y_all = df['Brix']

y_pred = loaded_model.predict(X_all)
print(y_pred, y_all)
