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

# Embarkedの補完
df_train.loc[df_train['PassengerId'].isin([62, 830]), 'Embarked'] = 'C'

# Fareの補完
df_test.loc[df_test['PassengerId'] == 1044, 'Fare'] = 13.675550

#Age変換のための関数
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
    # Ageの補完
    df['Age'] = df[['Age','Pclass']].apply(impute_age, axis = 1)

    # 性別の変換
    df['Sex'] = df['Sex'].map({"male": 0, "female": 1})

    # Embarked
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

df_train = pd.get_dummies(df_train, columns = ['Embarked'])
df_test = pd.get_dummies(df_test, columns = ['Embarked'])

data = [df_train, df_test]
for df in data:
    # Fareのカテゴリ変数化
    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)

    # Ageのカテゴリ変数化
    df.loc[ df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[ df['Age'] > 48, 'Age']  = 3
    df['Age'] = df['Age'].astype(int)

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
submission.to_csv('./result/tutorial5.csv', index=False)

### 特徴量の重要度の確認
forest.feature_importances_
for i,k in zip(X_train.columns,forest.feature_importances_):
  print(i,round(k,4))