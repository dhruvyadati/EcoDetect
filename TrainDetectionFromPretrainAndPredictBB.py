import supervision as sv
import torch
from deepforest import model
from deepforest import visualize
from deepforest import get_data
from deepforest import main
import pandas as pd
import cv2
import os
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def train_and_predict():
 
    cwd = os.getcwd()
    path = cwd + "/data/crops_data/uav_crops_data"
    csv_file ="testfile_multi.csv"
    csv_file_path = f"{path}/{csv_file}"
    df = pd.read_csv(get_data(csv_file_path))
    image_file="SOAP_061.png"
    image_file_path = f"{path}/{image_file}"
    root_dir = os.path.dirname(get_data(image_file_path))
    print("root_dir")
    print(root_dir)

    # Training the BoundigBox main model
    main_model_path = cwd + "/data/crops_data/uav_crops_data"
    main_model_csv_file_path = f"{main_model_path}/testfile_multi.csv"

    cfg = {}
    cfg["train"] = {}
    cfg["validation"] = {}
    cfg["num_classes"] = 5
    #cfg['nms_threshold'] = 0.05
    cfg["train"]["fast_dev_run"] = False
    cfg["batch_size"] = 2
    cfg["workers"] = 0
    cfg["train"]["epochs"] = 1
    cfg["train"]["csv_file"] = main_model_csv_file_path
    cfg["train"]["root_dir"] = main_model_path
    cfg["validation"]["csv_file"]=main_model_csv_file_path
    cfg["validation"]["root_dir"] = main_model_path
    
    # Create a new model with the appropriate number of classes 
    m = main.deepforest(config_args=cfg,
                        #config_args={"num_classes": 5},
                        num_classes = 5,
                        label_dict={
                            "jute": 0,
                            "maize": 1,
                            "rice": 2,
                            "sugarcane": 3,
                            "wheat": 4
                        })
 
    m.config["score_thresh"] = 0.4
  
    # Get a copy of the pretrained model so that we can use parts of it in our new model
    deepforest_release_model = main.deepforest()
    deepforest_release_model.use_release()

    # Replace the backbone of our new model with the pretrained backbone.
    m.model.backbone.load_state_dict(deepforest_release_model.model.backbone.state_dict())

    # Replace the regression head of the new model with the pretrained regression head.
    m.model.head.regression_head.load_state_dict(deepforest_release_model.model.head.regression_head.state_dict())
  
    m.trainer.fit(m)
    print ("Trainer fit done")
    m.trainer.validate(m)
    print ("Trainer validate done")

    global_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
   
    """
    # Save model weights directly
    model_save_path = f"{main_model_path}/EcoDetect_BB_{global_timestamp}.pt"
    m.save_model(model_save_path)
    print ("Model saved to: ")
    print(model_save_path)
    """

    # Call the predict_tile method with the crop_model
    print ("Calling predict tile")
    path = cwd + "/cropsimages/kagglecrops/test_crop_image"
    
    ext = [".jfif", ".jpg", ".png", ".jpeg"]
    for root, subdirs, files in os.walk(path):
        for file_name in files:
            # Check whether file is in text format or not
            if file_name.endswith(tuple(ext)):
                file_path = os.path.join(root, file_name)

                print("file_path")
                print (file_path)
                raster_path = get_data(file_path)
                print("raster_path")
                print (raster_path)

                iou_threshold = 0.15
                mosaic = True
                patch_size = 300
                patch_overlap = 0.05
                isBBMatched = False
                try: 
                    result = m.predict_tile(raster_path=raster_path,
                                            patch_size=patch_size,
                                            patch_overlap=patch_overlap,
                                            iou_threshold=iou_threshold,
                                            return_plot=False,
                                            mosaic=mosaic)
                    print ("Predict Tile done. result: ")
                    isBBMatched = True
                    print (result)
                except: 
                    print("Predict with bounding box dataset resulted in no matches")
 
                if (isBBMatched == False):
                    try: 
                        print ("Predicting match for the whole image")
                        result = m.predict_image(path=raster_path,
                                                return_plot=False)
                        print ("Predict Image done. result: ")
                        print (result)
                    except:
                        print("Predict image failed for")
                        print (raster_path)
    
    
                if result is not None: 
                    df = pd.DataFrame(result)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    scores_output_file = f"data/tiles_output_{timestamp}.csv"
                    #Append tile data to tiles_output_crops.csv
                    df.to_csv(scores_output_file, index=False, mode='a', header=False)
                    #Append numer of crops in this image to hist_num_<timestamp>.csv
                    histogram_output_file = f"data/hist_output_{global_timestamp}.csv"
                    hist_file = open(histogram_output_file, "a")  
                    line=file_path+","+str(len(df))
                    hist_file.write(line + '\n')
                    hist_file.close()
                    print ("Predict Tile for df done. result: ")
                    print (result)
                else:
                    print("No results to display")




def task_crop_model_train():
    train_and_predict()


# Fix a runtime error
if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    task_crop_model_train()  # execute this only when run directly, not when imported!

