import os
import pandas as pd
import tensorflow as tf
from Strategy_ml.src.utils import get_models_path

# data paths
# DATA_FOLDER = 'E:/urvish forex/intelligent-quantrader/train_test_data'
# MODELS_FOLDER = 'weights'
# FILE_NAME = 'AUD_USD_train.csv'
# CURRENCY_NAME = FILE_NAME.split("_train")[0]
# DATA_PATH = os.path.join(DATA_FOLDER, FILE_NAME)
BASE_PATH = "E:/urvish forex/forex-traading-algo"

MODELS_FOLDER = BASE_PATH + '/Strategy_ml/weights/'
DATA_FOLDER = BASE_PATH + '/train_test_data/'
# DATA_FOLDER = os.getcwd() +'\\train_test_data'
FILE_NAME = []
DATA_PATH = []
CURRENCY_NAME = []
TEST_DATA = []
TEST_DATA_PATH = []
for i in os.listdir(DATA_FOLDER):
    if "train" in i:
        FILE_NAME.append(i)
        CURRENCY_NAME.append(i.split("_train")[0])
        DATA_PATH.append(os.path.join(DATA_FOLDER, i))
    elif "unseen" in i:
        TEST_DATA.append(i)
        TEST_DATA_PATH.append(os.path.join(DATA_FOLDER, i))

SAVE_MATRIX_DETAIL = BASE_PATH + '/Strategy_ml/data/all_score_dict.json'

# test data
# TEST_DATA = 'AUD_USD_unseen.csv'
# TEST_DATA_PATH = os.path.join(DATA_FOLDER, TEST_DATA)
TEST_DROP_COLS = ['timestamp']

# target definition and columns to drop
TARGET = 'Position_final'

DROP_COLS = ['timestamp']

# features and indicators
INTERVALS = (5, 15, 30, 60, 100)

# path to save feature names for future use
feature_save_path = BASE_PATH + f'/Strategy_ml/data./feature_names_{TARGET}.pkl'

# feature selection threshold
fe_threshold = 0.3

# data splitting
SPLIT_DATE = pd.to_datetime(pd.Series(['2022/01/15']), format='%Y/%m/%d')

# model parameters
model_parameters = {'num_leaves': 2**7,
                    'learning_rate': 0.024,
                    'bagging_fraction': 0.7,
                    'bagging_freq': 10,
                    'feature_fraction': 0.4,
                    'feature_fraction_bynode': 0.8,
                    'min_data_in_leaf': 500,
                    'min_gain_to_split': 0.1,
                    'lambda_l1': 0.01,
                    'lambda_l2': 0,
                    'max_bin': 512,
                    'max_depth': -1,
                    'num_class':3,
                    'objective': 'multiclass',
                    'seed': None,
                    'feature_fraction_seed': None,
                    'bagging_seed': None,
                    'drop_seed': None,
                    'data_random_seed': None,
                    'boosting_type': 'gbdt',
                    'verbose': 1,
                    'metric': ['multi_logloss'],
                    'n_jobs': -1,
                    'force_col_wise': True,
                    }

# training parameters
fit_parameters = {
    'boosting_rounds': 2000,
    'early_stopping_rounds': 200,
    'verbose_eval': 500
}

# neg_samples_factor-
neg_samples_factor = 1

# previous context
n_context = 15

model_save_path = BASE_PATH + f'/Strategy_ml/{MODELS_FOLDER}/{TARGET}/lgb_{str(TARGET)}.txt'
saved_path = BASE_PATH + f'/Strategy_ml/{MODELS_FOLDER}/{TARGET}/lgb_{str(TARGET)}.txt'

# Use LSTM
use_lstm = False

# LSTM model config
lstm_config = {'optimizer': tf.keras.optimizers.SGD(learning_rate=0.01),
               'epochs': 10,
               'batch_size': 128,
               }

test_fe_names = ['MOM_100', 'RSI_5', 'AROON_up_60', 'AROON_down_15', 'AROONU_14', 'DMN_14', 'CCI_15', 'RSI_100',
                 'AROON_up_15', 'ULTOSC_30', 'WILLR_15', 'LINEARREG_ANGLE_60', 'BULLP_13', 'BEARP_13', 'AROONOSC_14',
                 'ULTOSC_15', 'AROON_down_30', 'LINEARREG_ANGLE_15', 'AROON_up_30', 'ROCP_60', 'MOM_60', 'ROCP_15',
                 'MOM_15', 'COPC_11_14_10', 'RSI_60', 'open_macdsignal', 'LINEARREG_ANGLE_30', 'high_macdsignal',
                 'low_macdsignal', 'close_macdsignal', 'ROCP_30', 'CCI_30', 'MOM_30', 'TSIs_13_25_13', 'WILLR_100',
                 'open_macd', 'WILLR_30', 'low_macd', 'high_macd', 'CCI_100', 'close_macd', 'RSI_15', 'RSI_30',
                 'WILLR_60', 'CCI_60', 'TSI_13_25_13']
# for blending models on test and validation data
ensemble = False

if ensemble:
    models_path_list = get_models_path(
        BASE_PATH + f'/Strategy_ml/{MODELS_FOLDER}/{TARGET}', [f'{TARGET}'])
mode = 'mean'

# models dir
PROD_MODELS_DIR = BASE_PATH + '/Strategy_ml/models'

