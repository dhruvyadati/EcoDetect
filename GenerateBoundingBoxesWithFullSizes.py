from PIL import Image
import os

# Calculate the full width and height of the images in a csv file
# Open the image
cwd = os.getcwd()
csv_path = cwd + "/data/crops_data/uav_crops_data/testfile_multi.csv"

file = open(csv_path, "a")  # append mode
line="line_num,image_path,label,croplabel,xmin,ymin,xmax,ymax\n"
file.writelines(line)

line_num = 1

def writeBoundingBoxesToCSV(image_dir_path, image_subdir_path):
    global line_num
    ext = [".jfif", ".jpg", ".png", ".jpeg"]
    for root, subdirs, files in os.walk(image_dir_path):
        for file_name in files:
            # Check whether file is in text format or not
            if file_name.endswith(tuple(ext)):
                file_path = os.path.join(root, file_name)
                
                dir = os.path.dirname(file_path)
                print (dir)
                base_dir = os.path.basename(dir)
                print(base_dir)

                image_path = image_subdir_path + "/" + base_dir + "/" + file_name
                print(image_path)

                croplabel = -1
                if (base_dir == "maize"):
                    croplabel = 1
                elif (base_dir == "rice"):
                    croplabel = 2
                elif (base_dir == "sugarcane"):
                    croplabel = 3
                elif (base_dir == "wheat"):
                    croplabel = 4
                elif (base_dir == "jute"):
                    croplabel = 0
                else:
                    print("Unsupported crop class. Exiting.")
                    exit

                image = Image.open(file_path)

                # Get the width and height
                width, height = image.size

                print("Width:", width)
                print("Height:", height)

                line = str(line_num) + "," + image_path + "," + str(base_dir) + "," + str(croplabel) + "," + str(0) + "," + str(0) + "," + str(width) + "," + str(height) + "\n"
                print(line)
                #file.write(line)
                line_num = line_num +1

image_basedir_path = cwd + "/data/crops_data/uav_crops_data/"
image_subdir_path = "kaggle/input/kag2"
image_dir_path = image_basedir_path + image_subdir_path 
writeBoundingBoxesToCSV(image_dir_path, image_subdir_path)
image_subdir_path = "kaggle/input/crop_images"
image_dir_path = image_basedir_path + image_subdir_path 
writeBoundingBoxesToCSV(image_dir_path, image_subdir_path)
image_subdir_path = "kaggle/input/some_more_images" 
image_dir_path = image_basedir_path + image_subdir_path 
writeBoundingBoxesToCSV(image_dir_path, image_subdir_path)

file.close()

