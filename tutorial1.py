import pandas as pd

# サンプルサブミットデータの読み込み
df_sample_submission = pd.read_csv("./all/gender_submission.csv")

# second.csvとして書き出し
df_sample_submission.to_csv("./result/tutorial1.csv",index=False)