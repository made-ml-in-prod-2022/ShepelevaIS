import os
from typing import List
import pytest

from py._path.local import LocalPath

from train_pipeline import *

from entities import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    TrainingParams,
    DownloadingParams
)


@pytest.fixture
def params():
    params = read_training_pipeline_params('./configs/train_config.yml')
    return params


def train_e2e(
    tmpdir: LocalPath,
    input_data_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
    features_to_drop: List[str],
    downloading_params
):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    params = TrainingPipelineParams(
        downloading_params=downloading_params,
        input_data_path=input_data_path,
        output_model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=2022),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
            features_to_drop=features_to_drop
        ),
        train_params=TrainingParams(model_type="LogisticRegression"),
    )
    real_model_path, metrics = train_pipeline(params)
    assert metrics["roc_auc"] > 0.5
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metric_path)


def test_train_e2e(params, tmpdir):
    train_e2e(
        tmpdir=tmpdir,
        input_data_path=params.input_data_path,
        categorical_features=params.feature_params.categorical_features,
        numerical_features=params.feature_params.numerical_features,
        target_col=params.feature_params.target_col,
        features_to_drop=params.feature_params.features_to_drop,
        downloading_params=params.downloading_params)
