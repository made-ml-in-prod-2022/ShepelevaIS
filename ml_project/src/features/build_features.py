import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from entities.feature_params import FeatureParams


from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Implementation of OrdinalEncoder for several columns"""
    def __init__(self):
        self.encoders = {}

    def fit(self, data):
        for i in range(data.shape[1]):
            self.encoders[i] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            self.encoders[i].fit(data[:, i].reshape(-1, 1))
        return self
        
    def transform(self, data):
        result = np.zeros(data.shape)
        for i in range(data.shape[1]):
            result[:, i] = self.encoders[i].transform(data[:, i].reshape(-1, 1)).reshape(1, -1)
        return result


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("encode", CategoricalEncoder()),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return categorical_pipeline


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:

    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df))


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
         ("scaler", StandardScaler())
         ]
    )
    return num_pipeline

def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ], 
        # remainder='passthrough'
    )
    return transformer


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df))


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target