# warningsを無視する
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
df_train = pd.read_csv("./all/train.csv")
df_test = pd.read_csv("./all/test.csv")

### 前処理

# 乗船した港(Embarked)の欠損値の補完(変更なし)
df_train.loc[df_train['PassengerId'].isin([62, 830]), 'Embarked'] = 'C'

# 運賃(Fare)の欠損値の補完(変更なし)
df_test.loc[df_test['PassengerId'] == 1044, 'Fare'] = 13.675550

# 年齢(Age)の欠損値の補完の変更
df_train.groupby('Pclass').mean()['Age']        # PclassごとにAgeの平均を算出

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 39
        elif Pclass == 2:
            return 30
        else:
            return 25
    else:
        return Age

data = [df_train, df_test]
for df in data:
    # 年齢(Age)の補完
    df['Age'] = df[['Age','Pclass']].apply(impute_age, axis = 1)

    # 性別(Sex)の変換
    df['Sex'] = df['Sex'].map({"male": 0, "female": 1})

    # 乗船した港(Embarked)の変換
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

df_train = pd.get_dummies(df_train, columns = ['Embarked'])
df_test = pd.get_dummies(df_test, columns = ['Embarked'])

df_train.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
df_test.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

X_train = df_train.drop(["PassengerId", "Survived"], axis=1) # 不要な列を削除
Y_train = df_train['Survived'] # Y_trainは、df_trainのSurvived列
X_test  = df_test.drop('PassengerId', axis=1).copy()

from sklearn.ensemble import RandomForestClassifier

# 学習と予測を行う
forest = RandomForestClassifier(random_state=1)
forest.fit(X_train, Y_train)
Y_prediction = forest.predict(X_test)
submission = pd.DataFrame({
        'PassengerId': df_test['PassengerId'],
        'Survived': Y_prediction
    })
submission.to_csv('./result/tutorial4.csv', index=False)