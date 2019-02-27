import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# データの読み込み
df_train = pd.read_csv("./all/train.csv")
df_test = pd.read_csv("./all/test.csv")
df_gender_submission = pd.read_csv("./all/gender_submission.csv")

### 前処理 ###

# 性別(Sex)の数値への変換
genders = {'male': 0, 'female': 1} # 辞書を作成
df_train['Sex'] = df_train['Sex'].map(genders)      # Sexをgendersを用いて変換
df_test['Sex'] = df_test['Sex'].map(genders)        # Sexをgendersを用いて変換

# 乗船した港(Embarked)の数値への変換
df_train = pd.get_dummies(df_train, columns=['Embarked'])   # ダミー変数化
df_test = pd.get_dummies(df_test, columns = ['Embarked'])   # ダミー変数化

# 使わない特徴量の列を削除
df_train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
print("-"*10)
df_train.head(6)
print("-"*10)

### 学習 ###

X_train = df_train.iloc[:,1:]
Y_train = df_train['Survived']

# 3分割交差検証を指定し、インスタンス化
kf = KFold(n_splits=3)

score_list = []

params = {
        'objective': 'binary',
        'metric' : 'binary_error',
        'learning_rate': 0.1,
        'num_leaves' : 300,
        'random_seed' : 1
}

for train_index, test_index in kf.split(X_train, Y_train):
    X_cv_train = X_train.iloc[train_index]
    X_cv_test = X_train.iloc[test_index]
    y_cv_train = Y_train[train_index]
    y_cv_test = Y_train[test_index]

    gbm = lgb.LGBMClassifier(objective='binary',
                        num_leaves = 300,
                        learning_rate = 0.1,
                        )
    gbm.fit(X_cv_train, y_cv_train,
            eval_set = [(X_cv_test, y_cv_test)],
            early_stopping_rounds=10,
            verbose=5)

    y_pred = gbm.predict(X_cv_test, num_iteration=gbm.best_iteration_)
    score_list.append(round(accuracy_score(y_cv_test,y_pred)*100,2))
    print(round(accuracy_score(y_cv_test,y_pred)*100,2))

score_list
gbm.get_params
#gbm.
df_gender_submission['Survived'] = y_pred
df_gender_submission.to_csv('./result/tutorial7.csv',index = False)


'''
X_train = df_train.iloc[:,1:]
Y_train = df_train['Survived']

# 3分割交差検証を指定し、インスタンス化
skf = KFold(n_splits=3)

params = {
        'objective': 'binary',
        'learning_rate': 0.1,
        'num_leaves' : 300
}


# skf.split(X_train.Ytrain)で、X_trainとY_trainを3分割し、交差検証をする
for train_index, test_index in skf.split(X_train, Y_train):
    X_cv_train = X_train.iloc[train_index]
    X_cv_test = X_train.iloc[test_index]
    y_cv_train = Y_train[train_index]
    y_cv_test = Y_train[test_index]

    lgb_train = lgb.Dataset(X_cv_train,y_cv_train)
    lgb_eval = lgb.Dataset(X_cv_test,y_cv_test)

    gbm = lgb.train(params = params,
            train_set = lgb_train,
            num_boost_round=50,
            valid_sets=lgb_eval,
            early_stopping_rounds=20,
            verbose_eval = 5)

    y_pred = gbm.predict(X_cv_test,num_iteration=gbm.best_iteration)

    # acuuracyを表示
    preds = np.round(gbm.predict(X_cv_test))
    print(round(accuracy_score(y_cv_test,preds)*100,2))

lgb_train = lgb.Dataset(X_train, Y_train)

# LightGBM のハイパーパラメータ
lgbm_params = {
    # ２値分類問題
    'objective': 'binary',
}

gbm = lgb.cv(params = lgbm_params,
            train_set = lgb_train,
            num_boost_round=50,
            nfold=5,
            stratified=False,
            early_stopping_rounds=None,
            verbose_eval = 5)

df_gender_submission['Survived'] = gbm
df_gender_submission.to_csv('./result/tutorial7.csv',index = False)
'''