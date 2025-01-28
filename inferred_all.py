
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
import spiceypy as spice
import scipy as sp
from scipy import ndimage, spatial
import matplotlib.pyplot as plt
import vicar
import skimage as skim
from skimage import morphology, feature, measure
import glob
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
import scipy
from photutils.detection import DAOStarFinder
import functions as fs
import math
from matplotlib.patches import Rectangle
import func_library_pipeline as flp
import yolo_inference_lib as yil
import json
import pickle

path_images = 'path/to/your/images/'
path_yolo = 'path/to/labels/from/yolov5/inference'


images = yil.load_images(path + '*.IMG', 32) #import the images
yil.exposure_duration(path, images) #select only the ones with an exposure duratio <= 2s
images8 = yil.filter_8bit_images(images) #select only the 8bit images
yolo_results = yil.upload_inferred_results(path_yolo) # import the results obtained after inferring yolo on the images

#this is the path where you stored the thresholds from the 'calcolo_backmean.py' code

file_path = '/lre/home/gquaglia/background_cassinifull/dic_thresh.json' 

with open(file_path, 'r') as json_file:
    thresh = json.load(json_file)

rp_final, stars, sat, gain_dic, energy = yil.inference_to_regionprops(yolo_results, images8_list, thresh)
# rp_final = dictionary containing for every image the regionprops objects (from scipy) in which there are many information about the cosmic rays in your images (area, centroids, orientation, eccentricity etc.)
# stars = same as for rp_final but for stars
# sat = same as for rp_final and stars but for satellites
# gain_dic = dictionary with the camera gain used for the images
# energy = dictionary with all the energies calculated for the cosmic rays
