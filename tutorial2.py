import pandas as pd

# サンプルサブミットデータの読み込み
df_sample_submission = pd.read_csv("./all/gender_submission.csv")

# サンプルサブミットデータのSurvived列を0に変更
df_sample_submission.loc[:, ["Survived"]] = 0

# second.csvとして書き出し
df_sample_submission.to_csv("./result/tutorial2.csv",index=False)