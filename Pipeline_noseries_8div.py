
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
import spiceypy as spice
import scipy as sp
from scipy import ndimage, spatial
import matplotlib.pyplot as plt
import glob
import vicar
import pandas as pd
from collections import defaultdict
import scipy
import yolo_inference_lib as yil

path_images = '/path/to/your/images/' # this is where you have downloaded the images from PDS Planetary Data System 

images = yil.load_images(path_images + '*.IMG', len(path_images)) 
yil.exposure_duration(path_images, images) 
images8 = yil.filter_8bit_images(images)
images8_dict, images8_list = yil.image_division(images8)

def image_to_png(images8_list, path):
    from PIL import Image
    for i, k in enumerate(images8_list):
        img = Image.fromarray(images8_list[k])
        img.save(path+str(k)+'.png')

# Change the path below with the path to the folder where you want to store the images for the network inference
image_to_png(images8_list, '/path/where/you/want/to/store/the/images/') 
