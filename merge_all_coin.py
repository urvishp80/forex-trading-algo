import glob
import pandas as pd

path = "E:/urvish forex/intelligent-quantrader/PREDICTED"
output_path = "E:/urvish forex/intelligent-quantrader/MERGED_COINS/"

s1_aud_usd = pd.read_csv("E:/urvish forex/intelligent-quantrader/PREDICTED/STRATEGY_1/M15/AUD_USD.csv")
s1_aud_usd = s1_aud_usd[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]

s1_bco_usd = pd.read_csv("E:/urvish forex/intelligent-quantrader/PREDICTED/STRATEGY_1/M15/BCO_USD.csv")
s1_bco_usd = s1_bco_usd[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]

s1_btc_usd = pd.read_csv("E:/urvish forex/intelligent-quantrader/PREDICTED/STRATEGY_1/M15/BTC_USD.csv")
s1_btc_usd = s1_btc_usd[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]

s1_de30_eur = pd.read_csv("E:/urvish forex/intelligent-quantrader/PREDICTED/STRATEGY_1/M15/DE30_EUR.csv")
s1_de30_eur = s1_de30_eur[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]

s1_eur_aud = pd.read_csv("E:/urvish forex/intelligent-quantrader/PREDICTED/STRATEGY_1/M15/EUR_AUD.csv")
s1_eur_aud = s1_eur_aud[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]

s1_eur_jpy = pd.read_csv("E:/urvish forex/intelligent-quantrader/PREDICTED/STRATEGY_1/M15/EUR_JPY.csv")
s1_eur_jpy = s1_eur_jpy[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]

s1_eur_usd = pd.read_csv("E:/urvish forex/intelligent-quantrader/PREDICTED/STRATEGY_1/M15/EUR_USD.csv")
s1_eur_usd = s1_eur_usd[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]

s1_gbp_jpy = pd.read_csv("E:/urvish forex/intelligent-quantrader/PREDICTED/STRATEGY_1/M15/GBP_JPY.csv")
s1_gbp_jpy = s1_gbp_jpy[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]

s1_gbp_usd = pd.read_csv("E:/urvish forex/intelligent-quantrader/PREDICTED/STRATEGY_1/M15/GBP_USD.csv")
s1_gbp_usd = s1_gbp_usd[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]

s1_nas100_usd = pd.read_csv("E:/urvish forex/intelligent-quantrader/PREDICTED/STRATEGY_1/M15/NAS100_USD.csv")
s1_nas100_usd = s1_nas100_usd[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]

s1_spx500_usd = pd.read_csv("E:/urvish forex/intelligent-quantrader/PREDICTED/STRATEGY_1/M15/SPX500_USD.csv")
s1_spx500_usd = s1_spx500_usd[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]

s1_us30_usd = pd.read_csv("E:/urvish forex/intelligent-quantrader/PREDICTED/STRATEGY_1/M15/US30_USD.csv")
s1_us30_usd = s1_us30_usd[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]

s1_usd_cad = pd.read_csv("E:/urvish forex/intelligent-quantrader/PREDICTED/STRATEGY_1/M15/USD_CAD.csv")
s1_usd_cad = s1_usd_cad[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]

s1_usd_jpy = pd.read_csv("E:/urvish forex/intelligent-quantrader/PREDICTED/STRATEGY_1/M15/USD_JPY.csv")
s1_usd_jpy = s1_usd_jpy[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]

s1_xau_usd = pd.read_csv("E:/urvish forex/intelligent-quantrader/PREDICTED/STRATEGY_1/M15/XAU_USD.csv")
s1_xau_usd = s1_xau_usd[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]


for count,i in enumerate(glob.glob(path + '/*/*/*.csv')):
    if "AUD_USD" in i and 'STRATEGY_1\M15' not in i:
        df = pd.read_csv(i)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]
        s1_aud_usd[f'Position_{count}'] = df['Position']
    elif "BCO_USD" in i and 'STRATEGY_1\M15' not in i:
        df = pd.read_csv(i)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]
        s1_bco_usd[f'Position_{count}'] = df['Position']
    elif "BTC_USD" in i and 'STRATEGY_1\M15' not in i:
        df = pd.read_csv(i)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]
        s1_btc_usd[f'Position_{count}'] = df['Position']
    elif "DE30_EUR" in i and 'STRATEGY_1\M15' not in i:
        df = pd.read_csv(i)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]
        s1_de30_eur[f'Position_{count}'] = df['Position']
    elif "EUR_AUD" in i and 'STRATEGY_1\M15' not in i:
        df = pd.read_csv(i)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]
        s1_eur_aud[f'Position_{count}'] = df['Position']
    elif "EUR_JPY" in i and 'STRATEGY_1\M15' not in i:
        df = pd.read_csv(i)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]
        s1_eur_jpy[f'Position_{count}'] = df['Position']
    elif "EUR_USD" in i and 'STRATEGY_1\M15' not in i:
        df = pd.read_csv(i)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]
        s1_eur_usd[f'Position_{count}'] = df['Position']
    elif "GBP_JPY" in i and 'STRATEGY_1\M15' not in i:
        df = pd.read_csv(i)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]
        s1_gbp_jpy[f'Position_{count}'] = df['Position']
    elif "GBP_USD" in i and 'STRATEGY_1\M15' not in i:
        df = pd.read_csv(i)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]
        s1_gbp_usd[f'Position_{count}'] = df['Position']
    elif "NAS100_USD" in i and 'STRATEGY_1\M15' not in i:
        df = pd.read_csv(i)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]
        s1_nas100_usd[f'Position_{count}'] = df['Position']
    elif "SPX500_USD" in i and 'STRATEGY_1\M15' not in i:
        df = pd.read_csv(i)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]
        s1_spx500_usd[f'Position_{count}'] = df['Position']
    elif "US30_USD" in i and 'STRATEGY_1\M15' not in i:
        df = pd.read_csv(i)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]
        s1_us30_usd[f'Position_{count}'] = df['Position']
    elif "USD_CAD" in i and 'STRATEGY_1\M15' not in i:
        df = pd.read_csv(i)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]
        s1_usd_cad[f'Position_{count}'] = df['Position']
    elif "USD_JPY" in i and 'STRATEGY_1\M15' not in i:
        df = pd.read_csv(i)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]
        s1_usd_jpy[f'Position_{count}'] = df['Position']
    elif "XAU_USD" in i and 'STRATEGY_1\M15' not in i:
        df = pd.read_csv(i)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Position']]
        s1_xau_usd[f'Position_{count}'] = df['Position']


s1_aud_usd.to_csv(output_path + "AUD_USD.csv", index=False)
s1_bco_usd.to_csv(output_path + "BCO_USD.csv", index=False)
s1_btc_usd.to_csv(output_path + "BTC_USD.csv", index=False)
s1_de30_eur.to_csv(output_path + "DE30_EUR.csv", index=False)
s1_eur_aud.to_csv(output_path + "EUR_AUD.csv", index=False)
s1_eur_jpy.to_csv(output_path + "EUR_JPY.csv", index=False)
s1_eur_usd.to_csv(output_path + "EUR_USD.csv", index=False)
s1_gbp_jpy.to_csv(output_path + "GBP_JPY.csv", index=False)
s1_gbp_usd.to_csv(output_path + "GBP_USD.csv", index=False)
s1_nas100_usd.to_csv(output_path + "NAS100_USD.csv", index=False)
s1_spx500_usd.to_csv(output_path + "SPX500_USD.csv", index=False)
s1_us30_usd.to_csv(output_path + "US30_USD.csv", index=False)
s1_usd_cad.to_csv(output_path + "USD_CAD.csv", index=False)
s1_usd_jpy.to_csv(output_path + "USD_JPY.csv", index=False)
s1_xau_usd.to_csv(output_path + "XAU_USD.csv", index=False)


print("process completed")
