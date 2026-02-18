import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from src.TALOS import logger
from src.TALOS.entity.config_entity import DataTransformationConfig
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.classes = ["plane"]

    def convert_vbox_to_yolo(self, size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        return (x * dw, y * dh, w * dw, h * dh)
    def initiate_data_transformation(self):
        try:
            source_dir = Path(self.config.data_path)
            target_root = Path(self.config.root_dir)
            split_map = {"train": "train", "test": "val"}

            for source_split, target_split in split_map.items():
                source_folder = source_dir / source_split
                img_target = target_root / "images" / target_split
                lbl_target = target_root / "labels" / target_split

                os.makedirs(img_target, exist_ok=True)
                os.makedirs(lbl_target, exist_ok=True)

                logger.info(f"Processing {source_split} split...")

                for file in os.listdir(source_folder):
                    if file.endswith(".xml"):
                        base_name = os.path.splitext(file)[0]
                        xml_path = source_folder / file
                        img_name = f"{base_name}.png"
                        img_path = source_folder / img_name
                        if os.path.exists(xml_path):
                            tree = ET.parse(xml_path)
                            root = tree.getroot()
                            size = root.find('size')
                            w = int(size.find('width').text)
                            h = int(size.find('height').text)
                            with open(lbl_target / f"{base_name}.txt", 'w') as f:
                                for obj in root.iter('object'):
                                    cls_name = obj.find('name').text
                                    if cls_name not in self.classes:
                                        continue

                                    cls_id = self.classes.index(cls_name)
                                    xmlbox = obj.find('bndbox')
                                    box = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                                           float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))

                                    yolo_box = self.convert_vbox_to_yolo((w, h), box)
                                    f.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in yolo_box])}\n")
                        if os.path.exists(img_path):
                            shutil.copy(img_path, img_target / img_name)

            logger.info("Transformation to YOLO structure complete.")

        except Exception as e:
            raise e