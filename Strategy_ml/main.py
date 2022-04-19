import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import time
import config
from src.dataset import get_data, get_features_targets, split_data, extract_most_important_features, \
    create_flatten_features, create_lstm_features, prepare_data
from src.logger import setup_logger
from src.models import BiLSTMModel, LightGBMModel, evaluate
from src.preprocess import get_processed_data, create_balanced_data


log = setup_logger(out_file=f'./logs/{config.TARGET}_training_{str(time.time())}.txt', stderr_level=logging.INFO)
log.info(f"{config.model_parameters}.")
scaler = StandardScaler()
le = LabelEncoder()

if __name__ == '__main__':
    for i,j in zip(config.CURRENCY_NAME, config.DATA_PATH):
        log.info(f"Starting data reading {i} currency.")
        df = get_data(j, drop_col=config.DROP_COLS)
        log.info(f"Spliting data for training and testing based on the date {config.SPLIT_DATE.iloc[0]} of {i} currency.")
        train_df, test_df = split_data(df, config.SPLIT_DATE.iloc[0])
        train_df['Position_final'] = le.fit_transform(train_df['Position_final'])
        log.info(f"saving label encoder for inference of {i} currency.")
        pickle.dump(le, open(f'./data/forex_{i}_le.pkl', 'wb'))
        test_df['Position_final'] = le.transform(test_df['Position_final'])
        print(f"train labels {np.unique(train_df['Position_final'], return_counts=True)}")
        print(f"testing labels {np.unique(test_df['Position_final'], return_counts=True)}")
        train_df = prepare_data(train_df)
        test_df = prepare_data(test_df)

        log.info(f"Performing features importance on full data of {i} currency.")
        features_names, _, corr_mtrx = extract_most_important_features(train_df)
        with open(f"./data/{i}_IMP_FEATURES.txt", 'w') as f:
            for s in features_names:
                f.write(str(s) + '\n')
        log.info(f"Important features are {features_names}. Total features: {len(features_names)} of {i} currency.")

        log.info(f"Count of target in training {train_df[config.TARGET].value_counts()} of {i} currency.")
        log.info(f"Count of target in testing {test_df[config.TARGET].value_counts()} of {i} currency.")

        log.info(f"Getting features and targets for training data of {i} currency.")
        features, targets = get_features_targets(train_df, config.TARGET, features_names, date_col='Date')
        log.info(f"Getting features and targets for testing data of {i} currency.")
        valid_feat, valid_targets = get_features_targets(test_df, config.TARGET, features_names, date_col='Date')

        log.info(f"Shape of train features: {features.shape}, Shape of train targets: {targets.shape} of {i} currency.")
        log.info(f"Shape of test features: {valid_feat.shape}, Shape of the test targets: {valid_targets.shape} of {i} currency.")

        if config.use_lstm:
            log.info(f"Normalizing features for LSTM model.")
            features = scaler.fit_transform(features[features_names])
            valid_feat = scaler.transform(valid_feat[features_names])

        if not config.use_lstm:
            features = features.values
            valid_feat = valid_feat.values
            targets = targets
            # for i in range(0, len(train_df), 150000):
            #     log.info(f"Preparing data for LGBM with previous context {config.n_context}.")
            #     # x = features[i:i+150000]
            #     # y = targets[i:i+150000]
            #     x, y = create_flatten_features(features[i:i+150000], targets[i:i+150000], config.n_context, features_names)
            #     np.save(f'./data/train_{i}.npy', x)
            #     np.save(f'./data/targets_{i}.npy', y)
            x, y = create_flatten_features(features, targets, config.n_context, features_names)
            # np.save(f'./data/train_features.npy', x)
            # np.save(f'./data/targets_targets.npy', y)
            valid_feat, valid_targets = create_flatten_features(valid_feat, valid_targets, config.n_context, features_names)
            # np.save('./data/valid_feat.npy', valid_feat)
            # np.save('./data/valid_targets.npy', valid_targets)
        else:
            log.info(f"Preparing data for LSTM with previous context {config.n_context}.")
            features, targets = create_lstm_features(features[0:200000], targets[0:200000], config.n_context, features_names)
            valid_feat, valid_targets = create_lstm_features(valid_feat, valid_targets, config.n_context, features_names)

        log.info(f"Shape of train features: {features.shape}, Shape of train targets: {targets.shape} of {i} currency.")
        log.info(f"Shape of test features: {valid_feat.shape}, Shape of the test targets: {valid_targets.shape} of {i} currency.")

        if config.use_lstm:
            log.info("Initializing LSTM model.")
            model = BiLSTMModel(input_shape=(config.n_context, features.shape[2]), output_shape=(1))
            model = model.create_model()

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                          loss=tf.keras.losses.BinaryCrossentropy(),
                          metrics=tf.keras.metrics.BinaryAccuracy())

            model.fit(features, targets, validation_data=(valid_feat, valid_targets), batch_size=config.lstm_config['batch_size'],
                      epochs=config.lstm_config['epochs'])
            model.evaluate(valid_feat, valid_targets)
            evaluate(model, valid_feat, valid_targets)
        else:
            log.info(f"Initializing LightGBM model for {i} currency.")
            model = LightGBMModel(config.model_parameters, config.fit_parameters, i, config)

            # valid_feat = np.load('./data/valid_feat.npy')
            # valid_targets = np.load('./data/valid_targets.npy')
            valid_feat = valid_feat
            valid_targets = valid_targets

            features = x
            targets = y

            print(features.shape, targets.shape)
            print(np.unique(targets, return_counts=True))
            print(np.unique(valid_targets, return_counts=True))

            model.train(features, targets, (valid_feat, valid_targets), save=False)

            acc_train, f1_train, precision_train, recall_train = model.eval(features, targets,"training")
            print(f"train results of {i}", acc_train, f1_train, precision_train, recall_train)
            acc, f1, precision, recall = model.eval(valid_feat, valid_targets, "validation")
            print(f"validation results of {i}", acc, f1, precision, recall)
            model.save()
            model_save_path = f'./{config.MODELS_FOLDER}/{config.TARGET}/lgb_{i}_{str(config.TARGET)}.txt'
            log.info(f"Saving features names to {model_save_path} for future use of {i} currency.")
