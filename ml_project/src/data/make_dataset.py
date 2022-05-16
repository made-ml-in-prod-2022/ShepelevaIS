from typing import Tuple, NoReturn

import pandas as pd
from boto3 import client
from sklearn.model_selection import train_test_split

from entities import DownloadingParams
from entities import SplittingParams


def download_data_from_s3(params: DownloadingParams) -> NoReturn:
    s3 = client("s3")
    s3.meta.download_file(params.s3_bucket, params.s3_path, params.output_folder)


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def train_valid_split(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    :rtype: object
    """
    if params.stratify_col:
      data_train, data_valid = train_test_split(
          data,
          stratify=data[params.stratify_col],
          test_size=params.val_size,
          random_state=params.random_state
      )
    else:
      data_train, data_valid = train_test_split(
          data,
          test_size=params.val_size,
          random_state=params.random_state
      )
    return data_train, data_valid
