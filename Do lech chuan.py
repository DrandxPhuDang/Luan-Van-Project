import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# nhap duong dan truy suat den thu muc chua cac du lieu cac dot do
path = r'D:\Luan Van Tot Nghiep\Data\Final_Data'
# nhap ten thu muc cua 1 dot do

namefolder = 'Data'
# nhap cot bat dau ve (cot chua buoc song)
startcol = 9
read_data = pd.read_csv(path + f'\{namefolder}.csv', sep=',')
df = pd.DataFrame(read_data)

Num1 = df[df['Number'] == 10]
brix1 = Num1['Acid']
print(brix1)
print(np.var(brix1))
print(np.std(brix1))