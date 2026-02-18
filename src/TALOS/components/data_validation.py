from src.TALOS.entity.config_entity import DataValidationConfig
import os
from src.TALOS import logger
class DataValidation:
    def __init__(self,config: DataValidationConfig):
        self.config = config
    def validate_all_files_exist(self):
        try:
            validation_status = True
            root_path = self.config.unzip_data_dir
            for folder in ["train", "test"]:
                folder_path = os.path.join(root_path, folder)
                if not os.path.exists(folder_path):
                    logger.error(f"Folder missing: {folder_path}")
                    print("Folder missing")
                    validation_status = False
                    continue
                all_files = os.listdir(folder_path)
                images = {os.path.splitext(f)[0] for f in all_files if f.endswith('.png')}
                xmls = {os.path.splitext(f)[0] for f in all_files if f.endswith('.xml')}
                missing_xml = images - xmls
                missing_png = xmls - images

                if missing_xml:
                    logger.error(f"In {folder}: Missing .xml for {missing_xml}")
                    validation_status = False

                if missing_png:
                    logger.error(f"In {folder}: Missing .png for {missing_png}")
                    validation_status = False
            os.makedirs(os.path.dirname(self.config.STATUS_FILE), exist_ok=True)
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation Status: {validation_status}")

            return validation_status
        except Exception as e:
            logger.error(f"Validation process failed: {e}")
            raise e