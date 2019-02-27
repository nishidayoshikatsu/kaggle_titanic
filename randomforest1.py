import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# データの読み込み
df_train = pd.read_csv('./all/train.csv')
df_test = pd.read_csv('./all/test.csv')

### 前処理 ###

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

     # FamilySizeとIsAloneの作成
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

# 不要な列の削除
df_train.drop(['Name', 'Cabin', 'Ticket','SibSp','Parch'], axis=1, inplace=True)
df_test.drop(['Name', 'Cabin', 'Ticket','SibSp','Parch'], axis=1, inplace=True)

### 学習 ###

# X_train、Y_train、X_testを作成
X_train = df_train.drop(["PassengerId","Survived"], axis=1)
Y_train = df_train["Survived"]
X_test  = df_test.drop("PassengerId", axis=1).copy()

# 学習
forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=25, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=15,
            min_weight_fraction_leaf=0.0, n_estimators=51,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
forest.fit(X_train, Y_train)
Y_pred = forest.predict(X_test)

# 提出データの作成
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred})
submission.to_csv('./result/randomforest1.csv', index=False)