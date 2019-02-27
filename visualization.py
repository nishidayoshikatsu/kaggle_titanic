#%%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
df_train = pd.read_csv("./all/train.csv")
df_test = pd.read_csv("./all/test.csv")
df_gender_submission = pd.read_csv("./all/gender_submission.csv")

# データの内容を可視化
print("df_train.head(5)" + str(df_train.head(5)))
print("df_test.head()" + str(df_test.head()))
print("df_train.shape" + str(df_train.shape))
print("df_test.shape" + str(df_test.shape))
print("df_gender_submission.shape" + str(df_gender_submission.shape))
# データ列の名前を可視化
print("df_train.columns" + str(df_train.columns))
print('-'*10)
print("df_test.columns" + str(df_test.columns))
print('-'*10)
# データの数・型を可視化
print("df_train.info()" + str(df_train.info()))
print('-'*10)
print("df_test.info()" + str(df_test.info()))
print('-'*10)
# 欠損値の数の可視化
print("df_train.isnull().sum():" + "\n" + str(df_train.isnull().sum()))
print('-'*10)
print("df_test.isnull().sum():" + "\n" + str(df_test.isnull().sum()))
print('-'*10)
# 要約統計量の可視化
df_full = pd.concat([df_train, df_test], axis = 0, ignore_index=True)
print("df_full.shape" + str(df_full.shape))
print("df_full.describe()" + str(df_full.describe()))
print("df_train.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9])" + str(df_train.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9])))
print("df_full.describe(include='O')" + str(df_full.describe(include='O')))
print('-'*10)

# 死亡者と生存者の可視化
plt.figure(1)
sns.countplot(x="Survived", data=df_train)
plt.title("count die or alive")         # plt.title("死亡者と生存者の数")
plt.xticks([0,1], ["die", "alive"])     # plt.xticks([0,1], ["死亡者", "生存者"])
# plt.show()
print("df_train['Survived'].value_counts():" + "\n" + str(df_train["Survived"].value_counts()))      # 死亡者と生存者数を表示
print("df_train['Survived'].value_counts()/len(df_train['Survived']):" + "\n" + str(df_train["Survived"].value_counts()/len(df_train["Survived"])))  # 死亡者と生存者割合を表示
print('-'*10)

# 男女別の死亡者と生存者数の可視化
plt.figure(2)
sns.countplot(x="Survived", hue="Sex", data=df_train)
plt.title("count die or alive in separated by gender")         # plt.title("男女別の死亡者と生存者の数")
plt.xticks([0.0,1.0], ["male", "female"])     # plt.xticks([0.0,1.0], ["male", "female"])
plt.title("count die or alive in separated by gender")         # plt.title("男女別の死亡者と生存者の数")
plt.legend(["die", "alive"])                # plt.legend(["死亡", "生存"])      # 調べる
# plt.show()
print("pd.crosstab(df_train['Sex'], df_train['Survived']):" + "\n" + str(pd.crosstab(df_train["Sex"], df_train["Survived"])))      # SexとSurvivedをクロス集計する
print("pd.crosstab(df_train['Sex'], df_train['Survived'], normalize='index'):" + "\n" + str(pd.crosstab(df_train["Sex"], df_train["Survived"], normalize="index")))  # クロス集計しSexごとに正規化する
print('-'*10)

# チケットクラス別の死亡者と生存者数の可視化
plt.figure(3)
sns.countplot(x="Pclass", hue="Survived", data=df_train)
plt.title("count die or alive in separated by ticket class")         # plt.title("チケットクラス別の死亡者と生存者の数")
plt.legend(["die", "alive"])                # plt.legend(["死亡", "生存"])      # 調べる
# plt.show()
print("pd.crosstab(df_train['Pclass'], df_train['Survived']):" + "\n" + str(pd.crosstab(df_train["Pclass"], df_train["Survived"])))      # PclassとSurvivedをクロス集計する
print("pd.crosstab(df_train['Pclass'], df_train['Survived'], normalize='index'):" + "\n" + str(pd.crosstab(df_train["Pclass"], df_train["Survived"], normalize="index")))  # クロス集計しPclassごとに正規化する
print('-'*10)

# 乗船者の年齢の分布
plt.figure(4)
sns.distplot(df_train["Age"].dropna(), kde=False, bins=30, label="全体")                            # 全体のヒストグラム
sns.distplot(df_train[df_train["Survived"] == 0].Age.dropna(), kde=False, bins=30, label="死亡")    # 死亡者のヒストグラム
sns.distplot(df_train[df_train["Survived"] == 1].Age.dropna(), kde=False, bins=30, label="生存")    # 生存者のヒストグラム
plt.title("Distribution of age of the persons on board")         # plt.title("乗船者の年齢の分布")
plt.legend(["all", "die", "alive"])
# plt.show()
df_train["CategoricalAge"] = pd.cut(df_train["Age"], 8)       # 年齢を8等分し、Categoricalという変数を作成
print("pd.crosstab(df_train['CategoricalAge'], df_train['Survived']):" + "\n" + str(pd.crosstab(df_train["CategoricalAge"], df_train["Survived"])))     # CategoricalAgeとSurvivedをクロス集計する
print("pd.crosstab(df_train['CategoricalAge'], df_train['Survived'], normalize='index'):" + "\n" + str(pd.crosstab(df_train["CategoricalAge"], df_train["Survived"], normalize="index")))   # クロス集計しCategoricalAgeごとに正規化する

# 同乗している兄弟・配偶者の数
plt.figure(5)
sns.countplot(x="SibSp", data=df_train)
plt.title("The number of siblings, spouse that passenger")         # plt.title("同乗している兄弟・配偶者の数")
# plt.show()

# 同乗している兄弟・配偶者の数別の死亡者と生存者の数の可視化
plt.figure(6)
df_train["SibSp_0_1_2over"] = [i if i <= 1 else 2 for i in df_train["SibSp"]]   # SibSpが0か1であればそのまま、2以上であれば2である特徴量SibSp_0_1_2overを作成
sns.countplot(x="SibSp_0_1_2over", hue="Survived", data=df_train)               # SibSp_0_1_2overごとに集計し、可視化
plt.legend(["die", "alive"])                # plt.legend(["死亡", "生存"])
plt.xticks([0,1,2], ["0 people","1 people","2 people over"])
plt.title("The number of survivors and the number by deaths of siblings, spouse that passenger")         # plt.title("同乗している兄弟・配偶者の数別の死亡者と生存者の数")
# plt.show()
print("pd.crosstab(df_train['SibSp_0_1_2over'], df_train['Survived']):" + "\n" + str(pd.crosstab(df_train["SibSp_0_1_2over"], df_train["Survived"])))     # SibSpとSurvivedをクロス集計する
print("pd.crosstab(df_train['SibSp_0_1_2over'], df_train['Survived'], normalize='index'):" + "\n" + str(pd.crosstab(df_train["SibSp_0_1_2over"], df_train["Survived"], normalize="index")))   # クロス集計しSibSpごとに正規化する

# 同乗している両親・子供の数の可視化
plt.figure(7)
sns.countplot(x="Parch", data=df_train)
plt.title("The number of parents and children that passenger")         # plt.title("同乗している両親・子供の数")
# plt.show()

# 同乗している両親・子供の数別の死亡者と生存者の数の可視化
plt.figure(8)
df_train["Parch_0_1_2_3over"] = [i if i <= 2 else 3 for i in df_train["Parch"]]   # Parchが2以下であればそのまま、3以上であれば3である特徴量Parch_0_1_2_3overを作成
sns.countplot(x="Parch_0_1_2_3over", hue="Survived", data=df_train)               # Parch_0_1_2_3overごとに集計し、可視化
plt.legend(["die", "alive"])                # plt.legend(["死亡", "生存"])
plt.xticks([0,1,2,3], ["0 people","1 people","2 people","3 people over"])
plt.title("The number of passengers to the number by the deaths of parents and children are survivors")         # plt.title("同乗している両親・子供の数別の死亡者と生存者の数")
# plt.show()
print("pd.crosstab(df_train['Parch_0_1_2_3over'], df_train['Survived']):" + "\n" + str(pd.crosstab(df_train["Parch_0_1_2_3over"], df_train["Survived"])))     # ParchとSurvivedをクロス集計する
print("pd.crosstab(df_train['Parch_0_1_2_3over'], df_train['Survived'], normalize='index'):" + "\n" + str(pd.crosstab(df_train["Parch_0_1_2_3over"], df_train["Survived"], normalize="index")))   # クロス集計しParchごとに正規化する

# 同乗している家族の人数別の死亡者と生存者の数の可視化
plt.figure(9)
df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"] + 1  # 同乗している家族の人数
df_train["lsAlone"] = 0
df_train.loc[df_train["FamilySize"] >= 2, "lsAlone"] = 1
sns.countplot(x="lsAlone", hue="Survived", data=df_train)               # lsAloneごとに集計し、可視化
plt.legend(["die", "alive"])                # plt.legend(["死亡", "生存"])
plt.xticks([0,1], ["1 people","2 people over"])
plt.title("The number of passengers to have a number of people by the deaths of family survivors")         # plt.title("同乗している家族の人数別の死亡者と生存者の数")
# plt.show()
print("pd.crosstab(df_train['lsAlone'], df_train['Survived']):" + "\n" + str(pd.crosstab(df_train["lsAlone"], df_train["Survived"])))     # lsAloneとSurvivedをクロス集計する
print("pd.crosstab(df_train['lsAlone'], df_train['Survived'], normalize='index'):" + "\n" + str(pd.crosstab(df_train["lsAlone"], df_train["Survived"], normalize="index")))   # クロス集計しlsAloneごとに正規化する

# 乗船者の運賃の分布
plt.figure(10)
sns.distplot(df_train["Fare"].dropna(), kde=False, hist=True)
plt.title("Distribution of persons on board of fare")     # plt.title("乗船者の運賃の分布")
plt.show()
df_train["CategoricalFare"] = pd.qcut(df_train["Fare"], 4)
df_train[["CategoricalFare", "Survived"]].groupby(["CategoricalFare"], as_index=False).mean()
print("pd.crosstab(df_train['CategoricalFare'], df_train['Survived']):" + "\n" + str(pd.crosstab(df_train["CategoricalFare"], df_train["Survived"])))     # CategoricalFareとSurvivedをクロス集計する
print("pd.crosstab(df_train['CategoricalFare'], df_train['Survived'], normalize='index'):" + "\n" + str(pd.crosstab(df_train["CategoricalFare"], df_train["Survived"], normalize="index")))   # クロス集計しCategoricalFareごとに正規化する

# 名前
print("df_test['Name'][0:5]" + str(df_test["Name"][0:5]))
set(df_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False))      # 敬称を抽出し、重複を省く
import collections
collections.Counter(df_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False))      # collections.Counterを使用して、数え上げる
df_train['Title'] = df_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)       # df_trainにTitle列を作成、Title列の値は敬称
df_test['Title'] = df_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)         # df_testにTitle列を作成、Title列の値は敬称
df_train.groupby('Title').mean()['Age']                                             # df_trainのTitle列の値ごとに平均値を算出

def title_to_num(title):        # 変換するための関数を作成
    if title == 'Master':
        return 1
    elif title == 'Miss':
        return 2
    elif title == 'Mr':
        return 3
    elif title == 'Mrs':
        return 4
    else:
        return 5

# リスト内包表記を用いて変換
df_train['Title_num'] = [title_to_num(i) for i in df_train['Title']]
df_test['Title_num'] = [title_to_num(i) for i in df_test['Title']]