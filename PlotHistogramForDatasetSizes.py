import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import os


plt.style.use('_mpl-gallery')
#df = pd.read_csv('data/hist_num_trees.csv') 
df = pd.read_csv('data/crops_data/uav_crops_data/testfile_multi.csv') 
    

# creating a histogram 
plt.xlabel("Crop Class - Control Group")
#plt.xlabel("Crop Class - Test Group")
plt.ylabel("Number Of Crop Images For Training")
x = ['sugarcane', 'rice', 'wheat', 'jute', 'maize']
plt.hist(df['label'], edgecolor='white', linewidth=1.2) 


plt.subplots_adjust(bottom=0.2) 
plt.subplots_adjust(left=0.2) 
plt.subplots_adjust(top=0.4) 
#plt.margins(y=0.2) 
plt.tight_layout()
plt.show() 

