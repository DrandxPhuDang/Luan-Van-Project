import numpy as np
import pandas as pd
from kennard_stone import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def features_selection(data, start_col, model=LinearRegression()):
    list_features = data.iloc[:0, start_col:]
    features = [f'{e}' for e in list_features]
    list_score = []
    list_score_copy = []
    df_score = pd.DataFrame()
    df_score_copy = pd.DataFrame()
    for i in features:
        X = data[i].values.reshape(-1, 1)
        y = data['Brix'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        R_train = np.corrcoef(y_train, y_pred_train, rowvar=False)
        list_score.append("{:.3f}".format(R_train[0][1]))

        R_Squared_train = r2_score(y_train, y_pred_train)
        list_score.append("{:.3f}".format(R_Squared_train))

        R_test = np.corrcoef(y_test, y_pred_test, rowvar=False)
        list_score.append("{:.3f}".format(R_test[0][1]))

        R_Squared_test = r2_score(y_test, y_pred_test)
        list_score.append("{:.3f}".format(R_Squared_test))

        R_train_ = float(list_score[0]) * 0.3
        R2_train_ = float(list_score[1]) * 0.7
        R_test_ = float(list_score[2]) * 0.3
        R2_test_ = float(list_score[3]) * 0.7

        All_train = R_train_ + R2_train_
        All_test = R_test_ + R2_test_

        list_score_copy.append(All_train)
        list_score_copy.append(All_test)

        list_score = pd.DataFrame(np.array(list_score), columns=[float(i)])
        df_score = pd.concat([df_score, list_score], axis=1)

        list_score_copy = pd.DataFrame(np.array(list_score_copy), columns=[float(i)])
        df_score_copy = pd.concat([df_score_copy, list_score_copy], axis=1)

        list_score = []
        list_score_copy = []

    df_score.insert(loc=0, column='Name_scores',
                    value=pd.DataFrame(np.array(['R_train', 'R_Squared_train', 'R_test', 'R_Squared_test'])))
    df_score_copy.insert(loc=0, column='Name_scores',
                         value=pd.DataFrame(np.array(['Score_Train', 'Score_Test'])))

    mean = 0
    cnt = 0
    list_mean_score = []
    for i in range(1, len(df_score['Name_scores']) + 1):
        for j in df_score.iloc[i - 1, 1:]:
            cnt += 1
            j = float(j)
            mean = j + mean
        mean = mean / cnt
        list_mean_score.append(mean)
        mean = 0
        cnt = 0

    r_train_ = list_mean_score[0] * 0.3
    r2_train_ = list_mean_score[1] * 0.7
    r_test_ = list_mean_score[2] * 0.3
    r2_test_ = list_mean_score[3] * 0.7
    list_mean_score_check = []
    Scores_train = r_train_ + r2_train_
    Scores_test = r_test_ + r2_test_
    list_mean_score_check.append(Scores_train)
    list_mean_score_check.append(Scores_test)

    df_score.insert(loc=1, column='Mean_scores',
                    value=pd.DataFrame(np.array(list_mean_score)))
    df_score.to_csv(r'/content/Data_selection.csv', index=False, header=True, na_rep='Unknown')
    df_score_copy.insert(loc=1, column='Mean_scores',
                         value=pd.DataFrame(np.array(list_mean_score_check)))
    df_score_copy.to_csv(r'/content/Cal_score_selection.csv', index=False, header=True, na_rep='Unknown')

    # --------------------- Kiem tra du lieu de loc buoc song-------------------------------------
    # df_train = df_score_copy[df_score_copy['Name_scores'] == 'Score_Train']
    # df_test = df_score_copy[df_score_copy['Name_scores'] == 'Score_Test']

    # mean_train = df_train['Mean_scores']
    # check_train = df_train.iloc[:1, 2:]
    # mean_test = df_test['Mean_scores']
    # check_test = df_test.iloc[:1, 2:]

    return df_score, Scores_train, Scores_test
