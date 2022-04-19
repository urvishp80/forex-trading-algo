import pandas as pd
import os

path = "E:/urvish forex/intelligent-quantrader/MERGED_COINS/"
date = "2022-03-16"
output_path = "E:/urvish forex/intelligent-quantrader/train_test_data/"


def split_df(df, date):
    train_data = df[df['timestamp'] <= date]
    unseen_data = df[df['timestamp'] > date]
    return train_data, unseen_data


for i in os.listdir(path):
    if "maxvote" in i:
        df = pd.read_csv(path + i)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position_final']]
        train_data, unseen_data = split_df(df, date)
        name = i.split("maxvote")[0]
        train_data.rename(columns={"Open":"open", "High":"high", "Low":"low", "Close":"close"}, inplace=True)
        unseen_data.rename(columns={"Open":"open", "High":"high", "Low":"low", "Close":"close"}, inplace=True)
        train_data.to_csv(output_path + f"{name}train.csv", mode='a', index=False)
        unseen_data.to_csv(output_path + f"{name}unseen.csv", mode='a', index=False)
