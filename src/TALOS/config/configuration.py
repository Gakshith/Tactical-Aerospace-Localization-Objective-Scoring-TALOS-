from src.TALOS.utils.common import read_yaml,create_directories
from src.TALOS.constants import CONFIG_FILE_PATH
from src.TALOS.entity.config_entity import DataIngestionConfig,DataTransformationConfig,DataValidationConfig,ModelTrainerConfig,ModelRunConfig
class ConfigurationManager:
    def __init__(self,config_filepath = CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)->DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])
        return DataValidationConfig(
            root_dir=config.root_dir,
            unzip_data_dir=config.unzip_data_dir,
            STATUS_FILE=config.STATUS_FILE,
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path
        )

    def get_model_train_config(self)->ModelTrainerConfig:
        config = self.config.model_train
        create_directories([config.root_dir])

        return ModelTrainerConfig(
            root_dir = config.root_dir,
            data_path=config.data_path,
            model_name = config.model_name,
            epochs = config.epochs,
            imgsz =  config.imgsz,
            rect =  config.rect,
            batch = config.batch,
            device = config.device,
            plots = config.plots
        )

    def get_model_run_config(self) -> ModelRunConfig:
        config = self.config.model_run
        create_directories([config.root_dir])
        return ModelRunConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            source_video_path=config.source_video_path,
            output_video_path=config.output_video_path,
        )