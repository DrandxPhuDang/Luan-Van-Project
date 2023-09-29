import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from kennard_stone import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


# split data set
url = f"D:\Luan Van Tot Nghiep\Data\Final_Data\Data_ALL.csv"
df = pd.read_csv(url)
listWavelegh = df.iloc[:0, 9:]
folder_bc = r"D:\Luan Van Tot Nghiep\Data\loc_waves\change_val_file_waves.csv"
listW = pd.read_csv(folder_bc)
features = [f'{e}' for e in listW]
X = df[features]
y = df['Brix']

# Creating the random forest regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Training the random forest regressor
regressor.fit(X_train, y_train)

# Making predictions on the test set
y_test_pred = regressor.predict(X_test)

# Evaluating the model
score = regressor.score(X_test, y_test)
print(f'R²: {score}')
print(f"Accuracy: {score * 100:.3f}%")
mse = mean_squared_error(y_test, y_test_pred)
print(f"Mean Squared Error: {mse}")

plt.scatter(y_test, y_test_pred)
plt.xlabel('Actual Response')
plt.ylabel('Predicted Response')
plt.title(f'Random Forest Regression (R²={score:.2f})')
reg = np.polyfit(y_test, y_test_pred, deg=1)
trend = np.polyval(reg, y_test)
plt.plot(y_test, trend, 'r')

plt.show()
