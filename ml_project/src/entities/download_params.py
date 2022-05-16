from dataclasses import dataclass, field


@dataclass()
class DownloadingParams:
    s3_bucket: str
    s3_path: str
    output_folder: str
