import sys
import os

from tqdm import tqdm
project_root = os.path.abspath(os.path.dirname(__file__))
submodule_path = os.path.join(project_root, 'pixelclip')
sys.path.insert(0, submodule_path)

import cv2
import numpy as np
import json
from PIL import Image

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

# CHANGE 2: Import the correct config function from the 'pixelclip' module
from pixelclip import add_pixelclip_config
from detectron2.projects.deeplab import add_deeplab_config

CONFIG_FILE = "configs/pixelclip_vit_base.yaml"
WEIGHTS_FILE = "weights/pixelclip_vit_base_sa1b/model_final.pth"

def setup_cfg(config_file, weights_file, custom_vocabulary):
    """
    Create a Detectron2 config object and set up the model for inference.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_pixelclip_config(cfg)
    cfg.merge_from_file(config_file)

    # --- Inject Custom Vocabulary ---
    metadata = MetadataCatalog.get("my_custom_dataset")
    metadata.stuff_classes = custom_vocabulary
    cfg.DATASETS.TRAIN = ("my_custom_dataset",)

    cfg.MODEL.WEIGHTS = weights_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.freeze()

    return cfg

def run_inference(image_path, vocabulary):
    print("Setting up the model configuration...")
    cfg = setup_cfg(CONFIG_FILE, WEIGHTS_FILE, vocabulary)

    print("Initializing the predictor...")
    predictor = DefaultPredictor(cfg)

    print(f"Loading image from: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    print("Running inference...")
    outputs = predictor(image_bgr)

    sem_seg_output = outputs["sem_seg"].to('cpu')

    prediction_map = sem_seg_output.argmax(dim=0).numpy()

    print("Inference complete. Visualizing results...")

    metadata = MetadataCatalog.get("my_custom_dataset")
    visualizer = Visualizer(image_bgr[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE_BW)
    segmented_image_vis = visualizer.draw_sem_seg(sem_seg_output.argmax(dim=0))

    segmented_image_bgr = segmented_image_vis.get_image()[:, :, ::-1]

    return prediction_map, segmented_image_bgr

def inference_pipeline():
    VOCABULARY_FILE = "datasets/ade150.json" 
    print(f"Loading vocabulary from {VOCABULARY_FILE}...")
    active_vocabulary = get_vocabulary_from_file(VOCABULARY_FILE)

    BASE_IMAGE = "../images"
    BASE_OUTPUT = "./output"
    for file_path in tqdm(os.listdir(BASE_IMAGE)):
        IMAGE_TO_TEST = os.path.join(BASE_IMAGE, file_path)
        prediction_map, visualized_output = run_inference(IMAGE_TO_TEST, active_vocabulary)
        found_class_indices = np.unique(prediction_map)
        found_class_names = [active_vocabulary[i] for i in found_class_indices]

        output_filename = os.path.join(BASE_OUTPUT, f"{file_path.split('.')[0]}.jpg")
        cv2.imwrite(output_filename, visualized_output)
        
        with open(os.path.join(BASE_OUTPUT,f"{file_path.split('.')[0]}.txt"), "w") as f:
            f.write('\n'.join(found_class_names))

       

def get_vocabulary_from_file(json_path):
    """Helper function to load class names from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data
    
def pred_to_class(prediction_map, active_vocabulary):
    found_class_indices = np.unique(prediction_map)
    found_class_names = [active_vocabulary[i] for i in found_class_indices]
    return found_class_names

def data_preparation():
    VOCABULARY_FILE = "datasets/ade150.json" 
    print(f"Loading vocabulary from {VOCABULARY_FILE}...")
    active_vocabulary = get_vocabulary_from_file(VOCABULARY_FILE)

    BASE_IMAGE = "../downloads"
    BASE_OUTPUT = "./train"

    os.makedirs(BASE_OUTPUT, exist_ok=True)

    for i, folder in tqdm(enumerate(os.listdir(BASE_IMAGE))):
        folder_path = os.path.join(BASE_IMAGE, folder, "images", "scene_cam_00_final_preview")
        image_1 = os.path.join(folder_path, "frame.0000.color.jpg")
        image_2 = os.path.join(folder_path, "frame.0001.color.jpg")
        
        scene = f"scene_{i}"

        OUTPUT_PATH = os.path.join(BASE_OUTPUT, scene)
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        prediction_map, visualized_output = run_inference(image_1, active_vocabulary)
        class_ = pred_to_class(prediction_map, active_vocabulary)

        output_filename_1 = os.path.join(OUTPUT_PATH, f"image0_pred.jpg")
        output_filename_2 = os.path.join(OUTPUT_PATH, f"image0_vis.jpg")
        output_filename_3 = os.path.join(OUTPUT_PATH, f"image0_class.txt")

        cv2.imwrite(output_filename_1, prediction_map) 
        cv2.imwrite(output_filename_2, visualized_output) 
        with open(output_filename_3, "w") as f:
            f.write(",".join(class_))

        prediction_map, visualized_output = run_inference(image_2, active_vocabulary)
        class_ = pred_to_class(prediction_map, active_vocabulary)

        output_filename_1 = os.path.join(OUTPUT_PATH, f"image1_pred.jpg")
        output_filename_2 = os.path.join(OUTPUT_PATH, f"image1_vis.jpg")
        output_filename_3 = os.path.join(OUTPUT_PATH, f"image1_class.txt")

        cv2.imwrite(output_filename_1, prediction_map) 
        cv2.imwrite(output_filename_2, visualized_output) 
        with open(output_filename_3, "w") as f:
            f.write(",".join(class_))

if __name__ == "__main__":
    data_preparation()