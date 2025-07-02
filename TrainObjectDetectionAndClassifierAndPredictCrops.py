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
import traceback
import sys

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def crop_model_train_and_predict_from_kaggle_crops():
 
    cwd = os.getcwd()
    crops_dir = cwd + "/data/crops_data/uav_crops_data/trained_cropsimages"
    
    # Training ResNet-50 Classification model for cropped images
    crop_model = model.CropModel(num_classes=5)

    path = cwd + "/data/crops_data/uav_crops_data"
    csv_file ="testfile_multi.csv"
    csv_file_path = f"{path}/{csv_file}"
    df = pd.read_csv(get_data(csv_file_path))
    try:
        boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    except:
        boxes = []
    image_file="SOAP_061.png"
    image_file_path = f"{path}/{image_file}"
    root_dir = os.path.dirname(get_data(image_file_path))
    print(root_dir)
    images = df.image_path.values
        
    print("############ Calling write_corps ############")
    crop_model.write_crops(boxes=boxes,
                            labels=df.label.values,
                           root_dir=root_dir,
                           images=images,
                           savedir=crops_dir)

    # Create a trainer
    # This one flips from deep training to one train. True will run fast. 
    # False will run the trainer for multiple batches and epochs
    crop_model.create_trainer(fast_dev_run=True, max_epochs=1)
    #crop_model.create_trainer(fast_dev_run=False, max_epochs=30)
    print("############ write_corps and create_trainer done ############")
    crop_model.load_from_disk(train_dir=crops_dir, val_dir=crops_dir)

    # Test training dataloader
    train_loader = crop_model.train_dataloader()
    assert isinstance(train_loader, torch.utils.data.DataLoader)

    # Test validation dataloader
    val_loader = crop_model.val_dataloader()
    assert isinstance(val_loader, torch.utils.data.DataLoader)

    # TRAIN & VALIDATE RestNet-50 classification model; 85/15 train/test split
    crop_model.trainer.fit(crop_model)
    crop_model.trainer.validate(crop_model)

    """
    true_class, predicted_class = crop_model.dataset_confusion(val_loader)
    print (true_class)
    print (predicted_class)

    # Display the confusion matrix for the Crop model. "Crop" here is a cropped image from 
    # bounding box of a larger image, not a plant crop.
    y_pred=np.argmax(predicted_class, axis=1)
    y_test=np.argmax(true_class, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()
    plt.close()
    """

    print("######## TRAINING and VALIDATE CROP_MODEL DONE ############")

    # Training the BoundingBox main model - RetinaNet object detection model
    
    main_model_path = cwd + "/data/crops_data/uav_crops_data"
    main_model_csv_file_path = f"{main_model_path}/testfile_multi.csv"

    cfg = {}
    cfg["train"] = {}
    cfg["validation"] = {}
    cfg["num_classes"] = 5
    cfg["train"]["fast_dev_run"] = False
    #cfg["train"]["fast_dev_run"] = True
    cfg["batch_size"] = 2
    #cfg["batch_size"] = 16
    #cfg["workers"] = 16
    cfg["workers"] = 0
    #cfg["train"]["epochs"] = 1
    cfg["train"]["epochs"] = 20
    cfg["train"]["csv_file"] = main_model_csv_file_path
    cfg["train"]["root_dir"] = main_model_path
    cfg["validation"]["csv_file"]=main_model_csv_file_path
    cfg["validation"]["root_dir"] = main_model_path
    m = main.deepforest(config_args=cfg,
                        num_classes = 5,
                        label_dict={
                            "jute": 0,
                            "maize": 1,
                            "rice": 2,
                            "sugarcane": 3,
                            "wheat": 4
                        })
                        #model=crop_model,
                        #existing_train_dataloader=train_loader,
                        #existing_val_dataloader=val_loader)
    
    m.config["score_thresh"] = 0.4
    m.create_trainer()
    #m.create_trainer(fast_dev_run=False, max_epochs=9)
    print ("Created trainer")

    # Train RetinaNet
    m.trainer.fit(m)
    m.trainer.validate(m)
    print ("############ Creating deepforest BB trainer done ############")

    global_file_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    """
    # Save model weights directly
    class_model_save_path = f"{main_model_path}/EcoDetect_Classification_{global_file_timestamp}.pt"
    m.save_model(class_model_save_path)
    print ("Classification Model saved to: ")
    print(class_model_save_path)

    # Save model weights directly
    detection_model_save_path = f"{main_model_path}/EcoDetect_BB_{global_file_timestamp}.pt"
    m.save_model(detection_model_save_path)
    print ("Detection Model saved to: ")
    print(detection_model_save_path)
    """

    # Call the predict_tile method with the integrated model
    print ("################### Calling predict tile")


    path = cwd + "/cropsimages/kagglecrops/test_crop_image"
   
    ext = [".jfif", ".jpg", ".png", ".jpeg"]
    for root, subdirs, files in os.walk(path):
        for file_name in files:
            # Check whether file is in text format or not
            if file_name.endswith(tuple(ext)):
                file_path = os.path.join(root, file_name)

                raster_path = get_data(file_path)
                iou_threshold = 0.15
                #return_plot = False
                mosaic = True
                #mosaic = False
                patch_size = 400
                patch_overlap = 0.05
                isBBMatched = False
                try: 
                    result = m.predict_tile(raster_path=raster_path,
                                            patch_size=patch_size,
                                            patch_overlap=patch_overlap,
                                            iou_threshold=iou_threshold,
                                            return_plot=False,
                                            mosaic=mosaic,
                                            crop_model=crop_model)
                    print ("######### Predict Tile done. result: ########")
                    isBBMatched = True
                    print (result)
                except Exception as e: 
                    print("Predict with bounding box dataset resulted in no matches")
                    traceback.print_exc(file=sys.stdout)
 
                if (isBBMatched == False):
                    try: 
                        print ("Predicting match for the whole image")
                        result = m.predict_image(path=raster_path,
                                                return_plot=False)
                        print ("######### Predict Image done. result: ########")
                        print (result)
                    except Exception as e:
                        print("Predict image failed for")
                        print (raster_path)
                        traceback.print_exc(file=sys.stdout)
    
    
                if result is not None: 
                    df = pd.DataFrame(result)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    scores_output_file = f"data/tiles_output_{timestamp}.csv"
                    #Append tile data to tiles_output_crops.csv
                    df.to_csv(scores_output_file, index=False, mode='a', header=False)
                    #Append numer of crops in this image to hist_num_<timestamp>.csv
                    histogram_output_file = f"data/hist_output_{global_file_timestamp}.csv"
                    hist_file = open(histogram_output_file, "a")  
                    line=file_path+","+str(len(df))
                    hist_file.write(line + '\n')
                    hist_file.close()
                    print ("######### Predict Tile for df done. result: ########")
                    print (result)
                else:
                    print("No results to display")




def task_crop_model_train():
    crop_model_train_and_predict_from_kaggle_crops()


# Fix a runtime error
if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    task_crop_model_train()  # execute this only when run directly, not when imported!

