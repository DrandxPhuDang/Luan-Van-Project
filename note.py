from Helper.Drand_Regression import DrandRgression
import pandas as pd


df = pd.read_csv(r'D:\Luan Van\Data\Final_Data\Random_measuring1.csv')
list_features = df.iloc[:0, 13:]
features = [f'{e}' for e in list_features]
X = df[features]
y = df['Brix']

# Create an instance of the Drand class
model = DrandRgression()

# Call the `model` and `predict` methods
model.fit(X, y)

y_pred = model.predict()

print(y_pred)
