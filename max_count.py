import os
import pandas as pd

path = "E:/urvish forex/intelligent-quantrader/MERGED_COINS/"

for i in os.listdir(path):
    if "AUD_USD" in i:
        df = pd.read_csv(path + i)
        droped_df = df.drop(['timestamp', 'Open', 'High', 'Low', 'Close'], axis=1)
        pos_data = []
        for j in droped_df.values:
            j = list(j)
            pos_data.append(max(j,key=j.count))
        df['Position_final'] = pos_data
        name = i.split('.')[0]
        df.to_csv(path + f"{name}_maxvote.csv", mode='a', index=False)
    elif "BCO_USD" in i:
        df = pd.read_csv(path + i)
        droped_df = df.drop(['timestamp', 'Open', 'High', 'Low', 'Close'], axis=1)
        pos_data = []
        for j in droped_df.values:
            j = list(j)
            pos_data.append(max(j, key=j.count))
        df['Position_final'] = pos_data
        name = i.split('.')[0]
        df.to_csv(path + f"{name}_maxvote.csv", mode='a', index=False)
    elif "BTC_USD" in i:
        df = pd.read_csv(path + i)
        droped_df = df.drop(['timestamp', 'Open', 'High', 'Low', 'Close'], axis=1)
        pos_data = []
        for j in droped_df.values:
            j = list(j)
            pos_data.append(max(j, key=j.count))
        df['Position_final'] = pos_data
        name = i.split('.')[0]
        df.to_csv(path + f"{name}_maxvote.csv", mode='a', index=False)
    elif "DE30_EUR" in i:
        df = pd.read_csv(path + i)
        droped_df = df.drop(['timestamp', 'Open', 'High', 'Low', 'Close'], axis=1)
        pos_data = []
        for j in droped_df.values:
            j = list(j)
            pos_data.append(max(j, key=j.count))
        df['Position_final'] = pos_data
        name = i.split('.')[0]
        df.to_csv(path + f"{name}_maxvote.csv", mode='a', index=False)
    elif "EUR_AUD" in i:
        df = pd.read_csv(path + i)
        droped_df = df.drop(['timestamp', 'Open', 'High', 'Low', 'Close'], axis=1)
        pos_data = []
        for j in droped_df.values:
            j = list(j)
            pos_data.append(max(j, key=j.count))
        df['Position_final'] = pos_data
        name = i.split('.')[0]
        df.to_csv(path + f"{name}_maxvote.csv", mode='a', index=False)
    elif "EUR_JPY" in i:
        df = pd.read_csv(path + i)
        droped_df = df.drop(['timestamp', 'Open', 'High', 'Low', 'Close'], axis=1)
        pos_data = []
        for j in droped_df.values:
            j = list(j)
            pos_data.append(max(j, key=j.count))
        df['Position_final'] = pos_data
        name = i.split('.')[0]
        df.to_csv(path + f"{name}_maxvote.csv", mode='a', index=False)
    elif "EUR_USD" in i:
        df = pd.read_csv(path + i)
        droped_df = df.drop(['timestamp', 'Open', 'High', 'Low', 'Close'], axis=1)
        pos_data = []
        for j in droped_df.values:
            j = list(j)
            pos_data.append(max(j, key=j.count))
        df['Position_final'] = pos_data
        name = i.split('.')[0]
        df.to_csv(path + f"{name}_maxvote.csv", mode='a', index=False)
    elif "GBP_JPY" in i:
        df = pd.read_csv(path + i)
        droped_df = df.drop(['timestamp', 'Open', 'High', 'Low', 'Close'], axis=1)
        pos_data = []
        for j in droped_df.values:
            j = list(j)
            pos_data.append(max(j, key=j.count))
        df['Position_final'] = pos_data
        name = i.split('.')[0]
        df.to_csv(path + f"{name}_maxvote.csv", mode='a', index=False)
    elif "GBP_USD" in i:
        df = pd.read_csv(path + i)
        droped_df = df.drop(['timestamp', 'Open', 'High', 'Low', 'Close'], axis=1)
        pos_data = []
        for j in droped_df.values:
            j = list(j)
            pos_data.append(max(j, key=j.count))
        df['Position_final'] = pos_data
        name = i.split('.')[0]
        df.to_csv(path + f"{name}_maxvote.csv", mode='a', index=False)
    elif "NAS100_USD" in i:
        df = pd.read_csv(path + i)
        droped_df = df.drop(['timestamp', 'Open', 'High', 'Low', 'Close'], axis=1)
        pos_data = []
        for j in droped_df.values:
            j = list(j)
            pos_data.append(max(j, key=j.count))
        df['Position_final'] = pos_data
        name = i.split('.')[0]
        df.to_csv(path + f"{name}_maxvote.csv", mode='a', index=False)
    elif "SPX500_USD" in i:
        df = pd.read_csv(path + i)
        droped_df = df.drop(['timestamp', 'Open', 'High', 'Low', 'Close'], axis=1)
        pos_data = []
        for j in droped_df.values:
            j = list(j)
            pos_data.append(max(j, key=j.count))
        df['Position_final'] = pos_data
        name = i.split('.')[0]
        df.to_csv(path + f"{name}_maxvote.csv", mode='a', index=False)
    elif "US30_USD" in i:
        df = pd.read_csv(path + i)
        droped_df = df.drop(['timestamp', 'Open', 'High', 'Low', 'Close'], axis=1)
        pos_data = []
        for j in droped_df.values:
            j = list(j)
            pos_data.append(max(j, key=j.count))
        df['Position_final'] = pos_data
        name = i.split('.')[0]
        df.to_csv(path + f"{name}_maxvote.csv", mode='a', index=False)
    elif "USD_CAD" in i:
        df = pd.read_csv(path + i)
        droped_df = df.drop(['timestamp', 'Open', 'High', 'Low', 'Close'], axis=1)
        pos_data = []
        for j in droped_df.values:
            j = list(j)
            pos_data.append(max(j, key=j.count))
        df['Position_final'] = pos_data
        name = i.split('.')[0]
        df.to_csv(path + f"{name}_maxvote.csv", mode='a', index=False)
    elif "USD_JPY" in i:
        df = pd.read_csv(path + i)
        droped_df = df.drop(['timestamp', 'Open', 'High', 'Low', 'Close'], axis=1)
        pos_data = []
        for j in droped_df.values:
            j = list(j)
            pos_data.append(max(j, key=j.count))
        df['Position_final'] = pos_data
        name = i.split('.')[0]
        df.to_csv(path + f"{name}_maxvote.csv", mode='a', index=False)
    elif "XAU_USD" in i:
        df = pd.read_csv(path + i)
        droped_df = df.drop(['timestamp', 'Open', 'High', 'Low', 'Close'], axis=1)
        pos_data = []
        for j in droped_df.values:
            j = list(j)
            pos_data.append(max(j, key=j.count))
        df['Position_final'] = pos_data
        name = i.split('.')[0]
        df.to_csv(path + f"{name}_maxvote.csv", mode='a', index=False)

