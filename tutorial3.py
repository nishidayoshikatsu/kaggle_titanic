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
df_gender_submission = pd.read_csv("./all/gender_submission.csv")

### 前処理

# 年齢(Age)の欠損値の補完
df_train["Age"].mean() # 年齢の平均値を算出
df_train['Age'] = df_train['Age'].fillna(30)    # 'Age'の欠損値に30を代入
df_test['Age'] = df_test['Age'].fillna(30)      # 'Age'の欠損値に30を代入

# 乗船した港(Embarked)の欠損値の補完
df_train[df_train['Embarked'].isnull()]         # df_trainでEmbarkedが欠損のデータを表示
df_train[df_train['Ticket'] == '113572']
df_test[df_test['Ticket'] == '113572']
df_train.loc[df_train['PassengerId'].isin([62, 830]), 'Embarked'] = 'C'     # 欠損値を'C'で埋め、表示して確認
df_train.loc[df_train['PassengerId'].isin([62, 830])]

# 運賃(Fare)の欠損値の補完
df_train[['Pclass','Fare']].groupby('Pclass').mean()    # PclassごとにFareの平均値を表示
df_test[df_test['Fare'].isnull()]                       # 欠損値があるレコードを確認
df_test.loc[df_test['PassengerId'] == 1044, 'Fare'] = 13.675550
df_test[df_test['PassengerId'] == 1044]
print('--df_trainの欠損値--')
print(df_train.isnull().sum()) # df_trainの欠損値を表示
print('-'*10 )
print('--df_testの欠損値--')
print(df_test.isnull().sum()) # df_testの欠損値を表示

# 性別(Sex)の数値への変換
genders = {'male': 0, 'female': 1} # 辞書を作成
df_train['Sex'] = df_train['Sex'].map(genders)      # Sexをgendersを用いて変換
df_test['Sex'] = df_test['Sex'].map(genders)        # Sexをgendersを用いて変換

# 乗船した港(Embarked)の数値への変換
df_train = pd.get_dummies(df_train, columns=['Embarked'])   # ダミー変数化
df_test = pd.get_dummies(df_test, columns = ['Embarked'])   # ダミー変数化

# 使わない特徴量の列を削除
df_train.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
df_test.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

### 学習

X_train = df_train.drop(["PassengerId", "Survived"], axis=1) # 不要な列を削除
Y_train = df_train['Survived'] # Y_trainは、df_trainのSurvived列
X_test  = df_test.drop('PassengerId', axis=1).copy()

from sklearn.ensemble import RandomForestClassifier

# ランダムフォレストのインスタンスを作成
forest = RandomForestClassifier(random_state=1)

# X_trainからY_trainを予測するように学習
forest.fit(X_train,Y_train)

# 正解率を表示
acc_log = round(forest.score(X_train, Y_train) * 100, 2)
print(round(acc_log,2,), '%')

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# 3分割交差検証を指定し、インスタンス化
kf = KFold(n_splits=3)

# skf.split(X_train.Ytrain)で、X_trainとY_trainを3分割し、交差検証をする
for train_index, test_index in kf.split(X_train, Y_train):
    X_cv_train = X_train.iloc[train_index]
    X_cv_test = X_train.iloc[test_index]
    y_cv_train = Y_train.iloc[train_index]
    y_cv_test = Y_train.iloc[test_index]
    forest = RandomForestClassifier(random_state=1)
    forest.fit(X_cv_train, y_cv_train) # 学習
    predictions = forest.predict(X_cv_test) # 予測
    # acuuracyを表示
    print(round(accuracy_score(y_cv_test,forest.predict(X_cv_test))*100,2))

# 学習と予測を行う
forest = RandomForestClassifier(random_state=1)
forest.fit(X_train, Y_train)
Y_prediction = forest.predict(X_test)
submission = pd.DataFrame({
        'PassengerId': df_test['PassengerId'],
        'Survived': Y_prediction
    })
submission.to_csv('./result/tutorial3.csv', index=False)