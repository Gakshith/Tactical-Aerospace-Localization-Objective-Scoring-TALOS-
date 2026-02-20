from src.TALOS.config.configuration import (ConfigurationManager)
from src.TALOS.components.model_run import ModelRun
from src.TALOS import logger


STAGE_NAME = "Model Running stage"

class ModelRunTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        model_run_config = config_manager.get_model_run_config()
        model_run = ModelRun(config=model_run_config)
        model_run.execute_run()
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelRunTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e