from src.TALOS.config.configuration import (ConfigurationManager)
from src.TALOS.components.model_train import ModelTrain
from src.TALOS import logger


STAGE_NAME = "Model Train stage"

class ModelTrainTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_train_config()
        model_trainer_config = ModelTrain(config=model_trainer_config)
        model_trainer_config.model_train()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e