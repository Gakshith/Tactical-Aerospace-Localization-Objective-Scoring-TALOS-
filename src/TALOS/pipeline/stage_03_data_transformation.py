from pathlib import Path
from src.TALOS.config.configuration import ConfigurationManager
from src.TALOS.components.data_transformation import DataTransformation
from src.TALOS import logger
STAGE_NAME = "Data_Transformation_Stage"
class DataTransformationPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"),"r") as f:
                status = f.read().split(" ")[-1]
            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(data_transformation_config)
                data_transformation.initiate_data_transformation()
        except Exception as e:
            raise e
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e









