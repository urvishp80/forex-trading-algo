from Strategy_ml.src.logger import LOGGER
from Strategy_ml.config import SAVE_MATRIX_DETAIL, BASE_PATH
import abc
import json
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, GRU, RNN, Embedding, Dropout, Flatten, Bidirectional, Conv1D, Add, Multiply, Input
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
import lightgbm as lgb
import numpy as np
import warnings
warnings.filterwarnings('ignore')

all_score_dict = {}


def multiclass_roc_auc_score(y_test, y_pred, average="weighted"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


class LightGBMModel:

    def __init__(self, model_parameters, fit_parameters,currency, config, inference=False):

        self.model_parameters = model_parameters
        self.fit_parameters = fit_parameters
        self.config = config
        self.logger = LOGGER
        self.inference = inference
        self.currency = currency

        if self.inference:
            self._model = self.load(f'{self.config.MODELS_FOLDER}\\{self.config.TARGET}\\lgb_{self.currency}_{str(self.config.TARGET)}.txt')
        else:
            self._model = None   # if self.config.saved_path is None else self.load(self.load(self.config.saved_path))

    def save(self):
        os.makedirs(f'{self.config.MODELS_FOLDER}\\{self.config.TARGET}\\', exist_ok=True)
        model_save_path = f'{self.config.MODELS_FOLDER}\\{self.config.TARGET}\\lgb_{self.currency}_{str(self.config.TARGET)}.txt'
        self._model.save_model(model_save_path)
        self.logger.info('Saved model to {}'.format(model_save_path))

    def load(self, path):
        model = lgb.Booster(model_file=path)  # init model
        self.logger.info('Loaded model from {}'.format(path))
        return model

    def train(self, features, targets, evaluation_set, seed=2022, save=True):
        trn = lgb.Dataset(features, label=targets)
        val = lgb.Dataset(evaluation_set[0], label=evaluation_set[1])

        self.model_parameters['seed'] = seed
        self.model_parameters['feature_fraction_seed'] = seed
        self.model_parameters['bagging_seed'] = seed
        self.model_parameters['drop_seed'] = seed
        self.model_parameters['data_random_seed'] = seed

        self._model = lgb.train(
            params=self.model_parameters,
            train_set=trn,
            valid_sets=[trn, val],
            num_boost_round=self.fit_parameters['boosting_rounds'],
            early_stopping_rounds=self.fit_parameters['early_stopping_rounds'],
            verbose_eval=self.fit_parameters['verbose_eval']
        )
        if save:
            self.save()

    def predict(self, features):
        preds = self._model.predict(features)
        return preds

    def eval(self, features, labels, type):
        preds_2d = self._model.predict(features)
        # preds_prob = self._model.predict_proba(features)
        # print(preds_prob.shape)
        # print(preds_2d.shape[1])
        preds = np.array([np.argmax(i) for i in preds_2d])
        preds = preds.reshape(-1,1)
        # print(f"value counts of prediction:- {np.unique(preds, return_counts=True)}")
        # print(f"value counts of actual label:- {np.unique(labels, return_counts=True)}")
        # print(labels.shape)
        # print(preds.shape)

        # preds = np.where(preds > 0.35, 1, 0)
        labels1 = np.arange(preds_2d.shape[1])
        auc = multiclass_roc_auc_score(labels, preds, average='weighted')
        # auc = roc_auc_score(labels, preds, multi_class='ovr', average='weighted')
        acc = accuracy_score(labels, np.round(preds))
        f1 = f1_score(labels, np.round(preds), average='weighted')
        precision = precision_score(labels, np.round(preds), average='weighted')
        recall = recall_score(labels, np.round(preds), average='weighted')

        all_score_dict[f"AUC score at the time of {type} of {self.currency} model is"] = round(auc,4)
        all_score_dict[f"Accuracy score at the time of {type}  of {self.currency} model is"] = round(acc,4)
        all_score_dict[f"F1 score at the time of {type} of {self.currency} model is"] = round(f1,4)
        all_score_dict[f"Precision score at the time of {type} of {self.currency} is"] = round(precision,4)
        all_score_dict[f"Recall score at the time of {type} of {self.currency} is"] = round(recall,4)

        with open(SAVE_MATRIX_DETAIL, 'w') as fp:
            json.dump(all_score_dict, fp)

        self.logger.info(f"AUC score at the time of {type} of {self.currency} model is {round(auc, 4)}.")
        self.logger.info(f"Accuracy score at the time of {type}  of {self.currency} model is {round(acc, 4)}.")
        self.logger.info(f"F1 score at the time of {type} of {self.currency} model is {round(f1, 4)}.")
        self.logger.info(f"Precision score at the time of {type} of {self.currency} is {round(precision, 4)}.")
        self.logger.info(f"Recall score at the time of {type} of {self.currency} is {round(recall, 4)}")
        return auc, f1, precision, recall


def evaluate(model, features, labels):
    preds = model.predict(features)
    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, np.round(preds))
    f1 = f1_score(labels, np.round(preds), average='binary')
    precision = precision_score(labels, np.round(preds), average='binary')
    recall = recall_score(labels, np.round(preds), average='binary')

    LOGGER.info(f"AUC score of model is {round(auc, 4)}.")
    LOGGER.info(f"Accuracy score of model is {round(acc, 4)}.")
    LOGGER.info(f"F1 score of model is {round(f1, 4)}.")
    LOGGER.info(f"Precision score is {round(precision, 4)}.")
    LOGGER.info(f"Recall score is {round(recall, 4)}")
    return preds, auc, f1, precision, recall


class BaseModel(abc.ABC):
    """Base class for models."""

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abc.abstractmethod
    def create_model(self):
        pass


class BiLSTMModel(BaseModel):
    """
     Bidirectional LSTM model class to create model.

     Arguments:
      * input_shape: Shape of the input tensor.
      * output_shape: Shape of the output tensor.
    """

    def __init__(self, input_shape, output_shape, *args, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape

        # model initialization
        self.model = None

    def create_model(self):

        inputs = tf.keras.Input(shape=self.input_shape)

        x = Bidirectional(
            LSTM(64, activation='elu', return_sequences=True))(inputs)
        x = Bidirectional(LSTM(64, activation='elu'))(x)

        x = Flatten()(x)
        x = Dense(64, activation='elu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='elu')(x)
        x = Dropout(0.3)(x)
        x = Dense(16, activation='elu')(x)
        output = Dense(self.output_shape, activation='sigmoid')(x)

        self.model = Model(inputs=inputs, outputs=output)
        return self.model
