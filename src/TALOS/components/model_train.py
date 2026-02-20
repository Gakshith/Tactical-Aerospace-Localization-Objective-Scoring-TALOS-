from ultralytics import YOLO
from src.TALOS.entity.config_entity import ModelTrainerConfig
from pathlib import Path
import os
import shutil
from src.TALOS import logger
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

    def move_trained_weights(self):
        try:
            source_path = Path("runs/detect/train3/weights/best.pt")
            target_dir = Path(self.config.root_dir)
            os.makedirs(target_dir, exist_ok=True)
            target_path = target_dir / "best.pt"
            if source_path.exists():
                shutil.move(str(source_path), str(target_path))
                logger.info(f"Successfully moved weights to: {target_path}")
                print(f"DONE: Model saved at {target_path}")
            else:
                logger.warning("best.pt not found! Check if training finished correctly.")
        except Exception as e:
            logger.error(f"Failed to move weights: {e}")