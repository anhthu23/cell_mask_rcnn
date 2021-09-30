from contextlib import nullcontext
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
 
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from modules.visualize import display_instances
from modules.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
 
import train
 
# %matplotlib inline 

def detect(model, images_path):
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    CELL_WEIGHTS_PATH = os.path.join(model)
    config = train.CustomConfig()
    CUSTOM_DIR = os.path.join(ROOT_DIR, "dataset","cell")
    
    # changes for inferencing.
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    
    config = InferenceConfig()
    config.display()
    
    # Device to load the neural network on.
    # Useful if you're training a model on the same 
    # machine, in which case use CPU and leave the
    # GPU for training.
    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
    
    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    # TODO: code for 'training' test mode not ready yet
    TEST_MODE = "inference"
    
    def get_ax(rows=1, cols=1, size=16):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.
        
        Adjust the size attribute to control how big to render images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax
    
    # Load validation dataset
    dataset = train.CustomDataset()
    dataset.load_custom_for_detect(images_path)
    
    # Must call before using the dataset
    dataset.prepare()
    
    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
    
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    
    weights_path = CELL_WEIGHTS_PATH 
    # weights_path = model.find_last()
    
    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    RESULTS_DIR = os.path.join(ROOT_DIR, "results/cell/")
    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Run object detection
    log('********************************* Erythrocyte and Leucocyte detection ***********************************')
    logfiles = ""
    for image_id in dataset.image_ids:
        image = dataset.load_image(image_id)
        results = model.detect([image], verbose=0)
        
        # Display results
        ax = get_ax(1)
        r = results[0]

        erythrocyteCount = 0
        leucocyteCount = 0
        print(len(r['masks']))
        print("-------------------------------------------------------------------------------------------")
        for class_id in r['class_ids']:
            if class_id == 1:
                erythrocyteCount = erythrocyteCount + 1
            if class_id == 2:
                leucocyteCount = leucocyteCount + 1

        display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    dataset.class_names, r['scores'], ax=ax,
                                    colors=[(0.4,0.3,0.1),(0.847,0.188,0.102),(0.984,0.571,0.933)],
                                    title=str(erythrocyteCount)+' erythrocytes and '+ str(leucocyteCount)+' leucocytes')

        log(' - '+ dataset.image_info[image_id]["id"] +': Dectected '+str(erythrocyteCount)+' erythrocytes and '+ str(leucocyteCount)+' leucocyte')
        logfiles += ' - '+ dataset.image_info[image_id]["id"] +': Dectected '+str(erythrocyteCount)+' erythrocytes and '+ str(leucocyteCount)+' leucocytes' +'\n'
        plt.savefig("{}/{}".format(submit_dir, dataset.image_info[image_id]["id"]))

    with open(submit_dir + '/detected-result.txt', 'a', encoding='utf-8') as f:
            f.write("Detected result:" + '\n')
            f.write("Weights: "+CELL_WEIGHTS_PATH + '\n')
            f.write(logfiles)
            f.write('\n')
    
############################################################
#  Detection
############################################################

if __name__ == '__main__':
    import argparse
 
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Detect erythrocyte and leucocyte')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'detect'")
    parser.add_argument('--images', required=False,
                        metavar="/path/to/detect/images/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    args = parser.parse_args()
 
    # Validate arguments
    if args.command == "detect":
        assert args.images, "Argument --images is required for detection"
        # assert args.weights, "Argument --weights is required for detection"
   
 
    print("images: ", args.images)
    print("weights: ", args.weights)
 
    weights = os.path.join(ROOT_DIR, "logs\object20210828T0156\mask_rcnn_object_0040.h5")
    if args.weights:
        weights = args.weights 
    if args.command.lower() == "detect":
        detect(weights, args.images)
    else:
        print("command not found: ", args.command)
        
    