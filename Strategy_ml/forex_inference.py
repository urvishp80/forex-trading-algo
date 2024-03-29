import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import time
import tensorflow as tf
import pickle

from Strategy_ml import config

from Strategy_ml.src.dataset import get_data, get_features_targets, merge_data, split_data, extract_most_important_features, get_lgbm_features, \
    create_flatten_features, create_lstm_features
from Strategy_ml.src.indicators import get_indicators, get_price_patterns, get_additional_indicators
from Strategy_ml.src.logger import setup_logger
from Strategy_ml.src.models import BiLSTMModel, LightGBMModel, evaluate


log = setup_logger(stderr_level=logging.INFO)
scaler = StandardScaler()

# for i in config.CURRENCY_NAME:
#     le_i = pickle.load(open(f"./data/forex_{i}_le.pkl", 'rb'))

def forex_inference(currency, test_data_path):
    log.info(f"Starting test data reading of {currency} currency..")
    df = get_data(test_data_path, drop_col=config.TEST_DROP_COLS)
    print(len(df))
    print("##########################", df)
    le_path = os.path.join(config.BASE_PATH, f"Strategy_ml\\data\\forex_{currency}_le.pkl")
    le = pickle.load(open(le_path, 'rb'))
    # df['Position_final'] = le.transform(df['Position_final'])
    # log.info(f"Getting indicator for data of {currency} currency.")
    # df = df.drop("Position_final", axis=1)
    df_indicators = get_indicators(df, intervals=config.INTERVALS)
    log.info(f"Getting price pattern for data of {currency} currency.")
    df_price_pattern = get_price_patterns(df)
    log.info(f"Getting additional indicators of {currency} currency.")
    df_add_indicators = get_additional_indicators(df)
    log.info(f"Merging all data into one of {currency} currency.")
    data = merge_data(df, df_indicators, df_price_pattern, df_add_indicators, test=True)
    print(data.head())
    print(len(data))
    imp_features_path = os.path.join(config.BASE_PATH, f"Strategy_ml\\data\\{currency}_IMP_FEATURES.txt")
    with open(imp_features_path, 'r') as f:
        fe_names = [line.rstrip('\n') for line in f]
    features_names = fe_names
    # features_names = config.test_fe_names
    log.info(f"Getting features testing data of {currency} currency.")
    # log.info(f"Getting features and targets for training data of {i} currency.")
    features, _ = get_features_targets(data, None, features_names, date_col='Date')
    log.info(f"Shape of test features: {features.shape} of of {currency} currency.")

    features = features.values
    features, _ = create_flatten_features(features, None, config.n_context, features_names)
    log.info(f"Shape of test features: {features.shape} of {currency} currency.")

    log.info(f"Initializing LightGBM model for {currency} currency.")
    model = LightGBMModel(config.model_parameters, config.fit_parameters, currency, config, inference=True)

    predictions = model.predict(features)
    log.info(f"Shape of the predictions {predictions.shape} of {currency} currency.")

    preds = [np.argmax(i) for i in predictions]
    pred_class = list(le.inverse_transform(preds))
    # label = df['Position_final'].to_list()
    # original_class = list(le.inverse_transform(label))

    df['pred_index'] = list(range(config.n_context + 1)) + preds
    df['pred_class'] = list(range(config.n_context + 1)) + pred_class
    # df['label_index'] = label
    # df['original_class'] = original_class

    df = df[["Date", "open", "high", "low", "close", "pred_index", "pred_class"]]
    # df = df[["Date", "open", "high", "low", "close", "label_index", "pred_index", "original_class", "pred_class"]]
    df_copy = df
    n = len(range(config.n_context + 1))
    df = df.iloc[n:, :]
    df.rename(columns={"Date":"timestamp"}, inplace=True)
    df_current = df.copy()
    df_current = df_current[["timestamp", "open", "high", "low", "close", "pred_class"]]
    df_current.rename(columns={"open":"Open", "high":"High", "low":"Low", "close":"Close", "pred_class":"Position"}, inplace=True)

    # df['Predictions'] = list(range(config.n_context)) + predictions.tolist()
    # df['Class'] = list(range(config.n_context)) + np.round(predictions).tolist()

    print(df.head())
    # df = df.sort_values(by='Date', ascending=False).reset_index(drop=True)
    os.makedirs(f'Strategy_ml/data/all_unseen_predictions_csv', exist_ok=True)
    os.makedirs(f'Strategy_ml/data/current_price_prediction', exist_ok=True)
    df_current.to_csv(f"Strategy_ml/data/current_price_prediction/predicted_{currency}.csv", index=False)
    # df.to_csv(f"Strategy_ml/data/all_unseen_predictions_csv/predicted_{currency}_{str(time.time())}.csv", index=False)
    df_current.to_csv(f"Strategy_ml/data/all_unseen_predictions_csv/predicted_{currency}_{str(time.time())}.csv", index=False)
    log.info(f"prediction saved of {currency} currency.")
    return None

if __name__ == '__main__':
    for i,j in zip(config.CURRENCY_NAME, config.TEST_DATA_PATH):
        forex_inference(i, j)
        # log.info(f"Starting test data reading of {i} currency..")
        # df = get_data(j, drop_col=config.TEST_DROP_COLS)
        # print(len(df))
        # le = pickle.load(open(f"./data/forex_{i}_le.pkl", 'rb'))
        # df['Position_final'] = le.transform(df['Position_final'])
        # log.info(f"Getting indicator for data of {i} currency.")
        # df_indicators = get_indicators(df, intervals=config.INTERVALS)
        # log.info(f"Getting price pattern for data of {i} currency.")
        # df_price_pattern = get_price_patterns(df)
        # log.info(f"Getting additional indicators of {i} currency.")
        # df_add_indicators = get_additional_indicators(df)
        # log.info(f"Merging all data into one of {i} currency.")
        # data = merge_data(df, df_indicators, df_price_pattern, df_add_indicators, test=True)
        # print(data.head())
        # print(len(data))
        # with open(f"./data/{i}_IMP_FEATURES.txt", 'r') as f:
        #     fe_names = [line.rstrip('\n') for line in f]
        # features_names = fe_names
        # # features_names = config.test_fe_names
        # log.info(f"Getting features testing data of {i} currency.")
        # # log.info(f"Getting features and targets for training data of {i} currency.")
        # features, _ = get_features_targets(data, None, features_names, date_col='Date')
        # log.info(f"Shape of test features: {features.shape} of of {i} currency.")
        #
        # features = features.values
        # features, _ = create_flatten_features(features, None, config.n_context, features_names)
        # log.info(f"Shape of test features: {features.shape} of {i} currency.")
        #
        # log.info(f"Initializing LightGBM model for {i} currency.")
        # model = LightGBMModel(config.model_parameters, config.fit_parameters, i, config, inference=True)
        #
        # predictions = model.predict(features)
        # log.info(f"Shape of the predictions {predictions.shape} of {i} currency.")
        #
        #
        # preds = [np.argmax(i) for i in predictions]
        # pred_class = list(le.inverse_transform(preds))
        # label = df['Position_final'].to_list()
        # original_class = list(le.inverse_transform(label))
        #
        # df['pred_index'] = list(range(config.n_context+1)) + preds
        # df['pred_class'] = list(range(config.n_context+1)) + pred_class
        # df['label_index'] = label
        # df['original_class'] = original_class
        #
        # df = df[["Date", "open", "high", "low", "close", "label_index", "pred_index", "original_class", "pred_class"]]
        # df_copy = df
        # n = len(range(config.n_context+1))
        # df = df.iloc[n:,:]
        #
        # # df['Predictions'] = list(range(config.n_context)) + predictions.tolist()
        # # df['Class'] = list(range(config.n_context)) + np.round(predictions).tolist()
        #
        # print(df.head())
        # # df = df.sort_values(by='Date', ascending=False).reset_index(drop=True)
        # os.makedirs(f'./data/all_unseen_predictions_csv', exist_ok=True)
        # df.to_csv(f"./data/all_unseen_predictions_csv/predicted_{i}_{str(time.time())}.csv", index=False)
        # log.info(f"prediction saved of {i} currency.")
