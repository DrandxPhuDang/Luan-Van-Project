from Helper.Drand_Regression import DrandRgression
from Helper.Drand_started_helper import print_score
import pandas as pd


df = pd.read_csv(r'D:\Luan Van\Data\Final_Data\Random_measuring.csv')
list_features = df.iloc[:0, 13:]
features = [f'{e}' for e in list_features]
X = df[features]
y = df['Brix']


import joblib

model = joblib.load(r'D:\Luan Van\model\PLS_regression.pkl')

y_pred = model.predict(X)
print(y_pred)
