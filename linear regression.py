import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from kennard_stone import train_test_split, KFold
from sklearn.model_selection import cross_val_score


# split data set
url = f'D:\Luan Van Tot Nghiep\Data\Final_Data\Data_All.csv'
folder_bc = r'D:\Luan Van Tot Nghiep\Data\loc_waves\change_val_file_waves.csv'
listW = pd.read_csv(folder_bc)
df = pd.read_csv(url)
listWavelegh = df.iloc[:0, 9:]
features = [f'{e}' for e in listW]
X = df[features]
y = df['Brix']
# print()
# print(y.shape)
wl = np.linspace(900, 1400, X.shape[1]) # wavelength
# X_train, X_test, y_train, y_test
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50, train_size=0.8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


model = LinearRegression()
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)


score = model.score(X_test, y_test)
print(f'R²: {score}')
print(f"Accuracy: {score*100:.3f}%")
mse = mean_squared_error(y_test, y_test_pred)
print(f"Mean Squared Error: {mse}")

plt.scatter(y_test, y_test_pred)
plt.xlabel('Actual Response')
plt.ylabel('Predicted Response')
plt.title(f'Linear Regression (R²={score:.2f})')
reg = np.polyfit(y_test, y_test_pred, deg=1)
trend = np.polyval(reg, y_test)
plt.plot(y_test, trend, 'r')

plt.show()