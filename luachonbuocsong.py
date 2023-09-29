import re
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")

# Chon tep data
Data_Train = pd.read_csv(r"D:\Luan Van Tot Nghiep\Data\Final_Data\Data_ALL.csv")

# Lay data tu cac cot buoc song
y_Train = Data_Train.Brix.values
# sua cot bat dau cua buoc song
X_Train = Data_Train.iloc[:, 9:].values
# print(y_Train)
# print(X_Train)
# sua cot bat dau cua buoc song
feature_list = list(Data_Train.iloc[:, 9:].columns)
# print(feature_list)

Xcal = X_Train
ycal = y_Train.ravel()
# print(Xcal)
# print(ycal)

randarray_coeff = pd.DataFrame()
randar = []
rtrain = []
rtest = []
oobscore = []


def np2pd(XX):
    df = pd.DataFrame()
    XX = XX.reshape(-1, 1)
    ii = 0
    a = []
    for i in XX:
        a = str(ii)
        df[a] = XX[ii, :]
        ii = ii + 1
        if ii == XX.shape[0]: ii = XX.shape[0] - 1
    return df


list_tb_cong = []
clfrand = RandomForestRegressor(n_estimators=100, oob_score=True)

print(f'Wait minutes...')


def value_train(data):
    global randarray_coeff
    for i in range(0, data):

        clfrand.fit(Xcal, ycal)
        randar = np2pd(clfrand.feature_importances_)
        n = 0
        products_list = randar.values.tolist()

        # tinh trung binh gia tri tai cac buoc song
        for v in products_list:
            answer = 0
            len_mylist = len(v)
            for n in v:
                answer += n
            answer = answer / len_mylist
            list_tb_cong.append([answer])

        randarray_coeff = randarray_coeff.append([randar], ignore_index=True)
    return randarray_coeff


# Chuyen data train thanh list, value_train(nhap vao so lan muon train)
list_randarray_coeff = value_train(100).values.tolist()


def changevalue(list_mean, list_value):
    ss = 0
    stt = 0
    so_sanh = []
    so_sanh_tong = []
    for mean in list_mean:
        mean = mean[0]
        # print(mean)
        for lst in range(stt, len(list_value)):
            c_list = list_value[lst]
            # print(c_list)
            for val in c_list:
                if (mean > val):
                    ss = 0
                    so_sanh.append(ss)
                if (mean < val):
                    ss = 1
                    so_sanh.append(ss)
            # print(so_sanh)
            so_sanh_tong.append(so_sanh)
            so_sanh = []
            break
        stt += 1
    so_sanh_tong = np.array(so_sanh_tong)
    return so_sanh_tong


# goi ham so sanh chuyen doi ve 0, 1
check_01 = changevalue(list_tb_cong, list_randarray_coeff)

# Chuyển mảng thành ma trận
check_01 = np.array(check_01)
check_01 = pd.DataFrame(check_01)

# Chuyển mảng thành ma trận
data_ref = np.array(randarray_coeff)
final_Data = pd.DataFrame(data_ref, columns=feature_list)

# Chuyển mảng thành ma trận
list_tb_cong = pd.DataFrame(list_tb_cong)
list_tb_cong = np.array(list_tb_cong)
list_tb_cong = pd.DataFrame(list_tb_cong)

# add waves value 0, 1
dt = 0
jj = len(feature_list)
list_re_waves = []
while jj > 0:
    jj = jj - 1
    val = check_01[jj]
    bc = feature_list[jj]
    bc = 'w: ' + bc
    list_re_waves.append(bc)
    final_Data.insert(loc=len(feature_list), column=bc, value=val)

final_Data.insert(loc=len(feature_list), column='Mean', value=list_tb_cong)


# Tim buoc song co gia tri tren 50%
def mathwavesimportant(list, data, number_run):
    val_X = 0
    sum_val_X = []
    loc_sum_val_X = []
    val_BC = []
    loc_val_BC = []
    for bc in list:
        val_X = 0
        for X in data[bc]:
            val_X += X
        sum_val_X.append([val_X])
        val_BC.append([bc])
        if val_X > number_run:
            loc_sum_val_X.append(val_X)
            loc_val_BC.append(bc)
    return sum_val_X, val_BC, loc_sum_val_X, loc_val_BC


# Goi ham tinh
find_im = mathwavesimportant(list_re_waves, final_Data, number_run=50)
# print(find_im[0])
# print(find_im[1])
# print(find_im[2])
# print(find_im[3])

# lay buoc song da loc
im_val = find_im[3]
len_bc = (len(im_val) - 1)

# loai ki tu
list_val_bc = []
while len_bc > 0:
    val_bc = im_val[len_bc]
    val_bc = val_bc[3:]
    list_val_bc.append(val_bc)
    len_bc = len_bc - 1

# change shape
list_val_bc = np.array(list_val_bc)
data_im_val = pd.DataFrame(list_val_bc)
data_im_val = data_im_val.T

data_im_val.to_csv(r"D:\Luan Van Tot Nghiep\Data\Final_Data\waves\buocsongdaloc.csv", index=False,
                   header=False, na_rep='Unknown')
# print(final_Data)
final_Data.to_csv(r"D:\Luan Van Tot Nghiep\Data\Final_Data\waves\luachonbuocsong.csv", index=False,
                  header=True, na_rep='Unknown')
