import os
import shutil
import kagglehub
from src.TALOS import logger
from src.TALOS.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            logger.info(f"Fetching {self.config.source_URL} via kagglehub...")
            downloaded_path = kagglehub.dataset_download(self.config.source_URL)
            os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
            if os.path.exists(downloaded_path):
                shutil.move(downloaded_path, self.config.local_data_file)
                logger.info(f"Dataset moved to: {self.config.local_data_file}")
        else:
            logger.info(f"Data already present at: {self.config.local_data_file}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        if self.config.local_data_file != unzip_path:
            if os.path.exists(self.config.local_data_file):
                logger.info(f"Syncing data to {unzip_path}")
                shutil.copytree(self.config.local_data_file, unzip_path, dirs_exist_ok=True)