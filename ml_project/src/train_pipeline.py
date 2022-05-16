import json
import logging
import sys

import click
import pandas as pd

from data import read_data, train_valid_split, download_data_from_s3
from entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from features import make_features
from features.build_features import extract_target, build_transformer
from models import (
    Classifier,
    model_validate
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    # download_data_from_s3(training_pipeline_params.downloading_params)
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    df_train, df_valid = train_valid_split(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"df_train.shape is {df_train.shape}")
    logger.info(f"df_valid.shape is {df_valid.shape}")

    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(df_train)
    features_train = make_features(transformer, df_train)
    target_train = extract_target(df_train, training_pipeline_params.feature_params)

    logger.info(f"features_train.shape is {features_train.shape}")
    classifier = Classifier(training_pipeline_params.train_params)

    classifier.fit(
        features_train, target_train
    )

    features_valid = make_features(transformer, df_valid)
    target_valid = extract_target(df_valid, training_pipeline_params.feature_params)

    features_valid_prepared = prepare_features_valid_for_predict(
        features_train, features_valid
    )

    logger.info(f"features_valid.shape is {features_valid_prepared.shape}")

    predictions = classifier.predict_proba(
        features_valid_prepared
    )[:, 1]

    metrics = model_validate(
        predictions,
        target_valid,
        training_pipeline_params.train_params.threshold
    )
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    path_to_model = classifier.save(training_pipeline_params.output_model_path)

    return path_to_model, metrics


def prepare_features_valid_for_predict(
    features_train: pd.DataFrame, features_valid: pd.DataFrame
):
    # small hack to work with categories
    features_train, features_valid = features_train.align(
        features_valid, join="left", axis=1
    )
    features_valid = features_valid.fillna(0)
    return features_valid


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
