from ultralytics import YOLO
from src.TALOS.entity.config_entity import ModelTrainerConfig

class ModelTrain:
    def __init__(self,config: ModelTrainerConfig):
        self.config = config
    def model_train(self):
        model = YOLO('yolo26s.pt')
        model.train(
        data=self.config.data_path,
        epochs=self.config.epochs,
        imgsz=self.config.imgsz,
        rect=self.config.rect,
        batch=self.config.batch,
        device=self.config.device,
        plots=self.config.plots
        )