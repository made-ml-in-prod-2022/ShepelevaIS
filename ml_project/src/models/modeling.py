import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from entities.train_params import TrainingParams

SklearnClassificationModel = Union[RandomForestClassifier, LogisticRegression]


def model_validate(
    predictions: np.ndarray, target: pd.Series, threshold: float
) -> Dict[str, float]:
    return {
        "roc_auc": roc_auc_score(target, predictions),
        "precision": precision_score(target, predictions > threshold),
        "recall": recall_score(target, predictions > threshold),
    }


class Classifier:
  def __init__(self, train_params: TrainingParams):
      self.train_params = train_params
      self.model = None

  def fit(self, features: pd.DataFrame, target: pd.Series) -> SklearnClassificationModel:
    if self.train_params.model_type == "RandomForestClassifier":
        self.model = RandomForestClassifier(
            n_estimators=100, random_state=self.train_params.random_state
        )
    elif self.train_params.model_type == "LogisticRegression":
        self.model = LogisticRegression()
    else:
        raise NotImplementedError()
    self.model.fit(features, target)
    return self

  def predict_proba(self, features: pd.DataFrame):
      predictions = self.model.predict_proba(features)
      return predictions

  def predict(self, features: pd.DataFrame):
      predictions = self.model.predict(features) >= self.train_params.threshold
      return predictions

  def save(self, output_file: str) -> str:
      with open(output_file, "wb") as f:
          pickle.dump(self.model, f)
      return output_file







# def model_train(
#     features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
# ) -> SklearnClassificationModel:
#     if train_params.model_type == "RandomForestClassifier":
#         model = RandomForestClassifier(
#             n_estimators=100, random_state=train_params.random_state
#         )
#     elif train_params.model_type == "LogisticRegression":
#         model = LogisticRegression()
#     else:
#         raise NotImplementedError()
#     model.fit(features, target)
#     return model


# def model_predict(
#     model: SklearnClassificationModel, features: pd.DataFrame,
# ) -> np.ndarray:
#     predictions = model.predict(features)
#     if use_log_trick:
#         predictions = np.exp(predictions)
#     return predictions





# def model_save(model: SklearnClassificationModel, output: str) -> str:
#     with open(output, "wb") as f:
#         pickle.dump(model, f)
#     return output