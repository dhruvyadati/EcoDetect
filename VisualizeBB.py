import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import supervision as sv
from deepforest import visualize
import numpy as np

def add_headers_to_csv_files():
    cwd = os.getcwd()
    # Note: Move the csv files to the specific output directory before reading the files
    tiles_output_dir = cwd + "/data/tiles_output"
    
    for root,dirs,files in os.walk(tiles_output_dir):
        for f in files:
            if f.endswith(".csv"):
                file_name = f"{tiles_output_dir}/{f}"
                print("\nOriginal file:") 
                print(file_name) 
                file = pd.read_csv(file_name) 
                print("\nOriginal file:") 
                print(file_name) 
                
                # adding header 
                headerList = ['xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_path', 'cropmodel_label', 'cropmodel_score'] 
                
                # converting data frame to csv 
                file.to_csv(file_name, header=headerList, index=False) 
                
                # display modified csv file 
                file2 = pd.read_csv(file_name) 
                print('\nModified file:') 
                print(file2) 

def add_headers_to_pretrained_csv_files():
    cwd = os.getcwd()
    tiles_output_dir = cwd + "/data/tiles_output"
    
    for root,dirs,files in os.walk(tiles_output_dir):
        for f in files:
            if f.endswith(".csv"):
                file_name = f"{tiles_output_dir}/{f}"
                print("\nOriginal file name:") 
                print(file_name)
                file = pd.read_csv(file_name) 
                print("\nOriginal file:") 
                print(file) 
                
                # adding header 
                headerList = ['xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_path'] 
                
                # converting data frame to csv 
                file.to_csv(file_name, header=headerList, index=False) 
                
                # display modified csv file 
                file2 = pd.read_csv(file_name) 
                print('\nModified file:') 
                print(file2) 


def convert_to_sv_format(df):
    """
    Modified version of original method in deepforest visualize.py to incorporate cropmodel labels
    Convert DeepForest prediction results to a supervision Detections object.
    """

    print("df:")
    print(df)
    print ("------------")

    # Extract bounding boxes as a 2D numpy array with shape (_, 4)
    boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(np.float32)

    label_to_numeric_dict={"jute": 0,
                "maize": 1,
                "rice": 2,
                "sugarcane": 3,
                 "wheat": 4}
    
    numeric_to_label_dict={0: "jute",
                           1: "maize",
                           2: "rice",
                           3: "sugarcane",
                           4: "wheat"}
    

    # Extract labels as a numpy array
    labels = df['cropmodel_label'].values.astype(int)
    print("labels: ")
    print(labels)
    print ("------------")


    df["cropmodel_label"] = df.cropmodel_label.apply(
                lambda x: numeric_to_label_dict[x])
    print ("df[cropmodel_label]: " + df["cropmodel_label"])
    print ("------------")
    
    # Extract scores as a numpy array
    scores = np.array(df['cropmodel_score'].tolist())
    print("scores: " )
    print(scores)
    print ("------------")
    # Create a reverse mapping from integer to string labels
    class_name = {v: k for k, v in label_to_numeric_dict.items()}
    print("class_name: ")
    print(class_name)
    print ("------------")

    return sv.Detections(
        xyxy=boxes,
        class_id=labels,
        confidence=scores,
        data={"class_name": [class_name[class_id] for class_id in labels]})


def visualize_bounding_boxes_pretrained():

    cwd = os.getcwd()
    tiles_output_dir = cwd + "/data/tiles_output"
    image_path_dir = cwd + "/cropsimages/kagglecrops/test_crop_image"
    
    for root,dirs,files in os.walk(tiles_output_dir):
        for f in files:
            if f.endswith(".csv"):
                file_name = f"{tiles_output_dir}/{f}"
                df = pd.read_csv(file_name)

                # Check that dataframe has at least one row
                if df is not None and len(df['image_path']) != 0: 
                                print ("Displaying boxes with labels for tiles_output: " + f)
                                # Convert custom prediction results to supervision format
                                sv_detections = visualize.convert_to_sv_format(df)
                                image_name = df['image_path'].unique()[0]
                                image_path = f"{image_path_dir}/{image_name}"
                                print("Image: " + image_path)
                                image = cv2.imread(image_path)

                                # Visualize predicted bounding boxes
                                bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=5)
                                annotated_frame = bounding_box_annotator.annotate(
                                    scene=image.copy(),
                                    detections=sv_detections,
                                    
                                )
                                # Display the image using Matplotlib
                                plt.imshow(annotated_frame)
                                plt.axis('off')  # Hide axes for a cleaner look
                                plt.show()

                                # Visualize predicted bounding boxes with labels           
                                label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER,
                                                                    text_color=sv.Color.BLACK, 
                                                                    text_scale=0.5, 
                                                                    text_thickness=1, 
                                                                    text_padding=10, 
                                                                    color_lookup=sv.ColorLookup.CLASS)
                                annotated_frame = label_annotator.annotate(
                                    scene=image.copy(),
                                    detections=sv_detections,
                                    labels=sv_detections['class_name'])

                                # Display the image using Matplotlib
                                plt.imshow(annotated_frame)
                                plt.axis('off')  # Hide axes for a cleaner look
                                plt.show()


def visualize_bounding_boxes():

    cwd = os.getcwd()
    tiles_output_dir = cwd + "/data/tiles_output"
    image_path_dir = cwd + "/cropsimages/kagglecrops/test_crop_image"
    
    for root,dirs,files in os.walk(tiles_output_dir):
        for f in files:
            if f.endswith(".csv"):
                file_name = f"{tiles_output_dir}/{f}"
                df = pd.read_csv(file_name)

                # Check that dataframe has at least one row
                if df is not None and len(df['image_path']) != 0: 
                                print ("Displaying boxes with labels for tiles_output: " + f)
                                # Convert custom prediction results to supervision format
                                sv_detections = convert_to_sv_format(df)
                                image_name = df['image_path'].unique()[0]
                                image_path = f"{image_path_dir}/{image_name}"
                                print("Image: " + image_path)
                                image = cv2.imread(image_path)

                                # Visualize predicted bounding boxes
                                #color_palette = sv.ColorPalette.from_hex(['#ff0000', '#00ff00', '#0000ff'])
                                #color_palette.by_idx(1)
                                color_palette = sv.ColorPalette.from_hex(['#0000ff'])
                                # Color(r=0, g=255, b=0)
                                bounding_box_annotator = sv.BoundingBoxAnnotator(color=color_palette,
                                                                                 #color_lookup=sv.ColorLookup.CLASS,
                                                                                 thickness=5)
                                annotated_frame = bounding_box_annotator.annotate(
                                    scene=image.copy(),
                                    detections=sv_detections,
                                    
                                )
                                # Display the image using Matplotlib
                                plt.imshow(annotated_frame)
                                plt.axis('off')  # Hide axes for a cleaner look
                                plt.show()

                                # Visualize predicted bounding boxes with labels
                                color_palette = sv.ColorPalette.from_hex(['#ff0000', '#00ff00', '#00fff7', '#fff200', '#ff00f2'])
                                color_palette.by_idx(1)
                                # Color(r=0, g=255, b=0)
                
                                label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER, 
                                                                    color=color_palette,  
                                                                    text_color=sv.Color.BLACK, 
                                                                    text_scale=0.5, 
                                                                    text_thickness=1, 
                                                                    text_padding=10, 
                                                                    color_lookup=sv.ColorLookup.CLASS)
                                annotated_frame = label_annotator.annotate(
                                    scene=image.copy(),
                                    detections=sv_detections,
                                    labels=sv_detections['class_name'])

                                # Display the image using Matplotlib
                                plt.imshow(annotated_frame)
                                plt.axis('off')  # Hide axes for a cleaner look
                                plt.show()


# NOTE: Uncomment and run one of the below one time to modify the tiles output 
# files to add headers depending on whether it is the classification model 
# or the pretrained model.
#add_headers_to_csv_files()
#add_headers_to_pretrained_csv_files()

# Uncomment and run one of the below after adding headers 
visualize_bounding_boxes()
#visualize_bounding_boxes_pretrained()
