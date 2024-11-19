
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

images = yil.load_images('/lre/home/gquaglia/cassini_full/*.IMG', 32)
yil.exposure_duration('/lre/home/gquaglia/cassini_full', images)
images8 = yil.filter_8bit_images(images)
distance_qmpf, time, latitude, xyz_IAU, distance_enceladus, xyz_ence =yil.distance('/lre/home/gquaglia/cassini_full', 'Enceladus', images8)
images8_dict, images8_list = yil.image_division(images8)
yolo_results = yil.upload_inferred_results('/lre/home/gquaglia/env/yolov5/runs/detect/exp9_cassini_full/labels')


# Specifica il percorso e il nome del file
file_path = '/lre/home/gquaglia/background_cassinifull/dic_mean.json'

# Carica il dizionario dal file JSON
with open(file_path, 'r') as json_file:
    mean_dict = json.load(json_file)

file_path = '/lre/home/gquaglia/background_cassinifull/dic_std.json'

# Carica il dizionario dal file JSON
with open(file_path, 'r') as json_file:
    std_dict = json.load(json_file)

# Specifica il percorso e il nome del file
file_path = '/lre/home/gquaglia/background_cassinifull/dic_thresh.json'

# Carica il dizionario dal file JSON
with open(file_path, 'r') as json_file:
    thresh = json.load(json_file)

rp_final, stars, sat, gain_dic, energy = yil.inference_to_regionprops(yolo_results, images8_list, thresh)
df_final, df_dist, df_time, df_lat = yil.build_dataframe(xyz_IAU, rp_final, latitude, distance_qmpf, time, images8_dict, thresh, energy,distance_enceladus, images8_list, thresh)
#df_final.to_pickle('/lre/home/gquaglia/background_cassinifull/dataframe_cassinifull1')
#import joblib
#joblib.dump(rp_final, '/lre/home/gquaglia/background_cassinifull/cosmics2.pkl')
#with open('/lre/home/gquaglia/background_cassinifull/energy_dictionary_backmedian.pkl', 'wb') as f:
   #pickle.dump(energy, f)
#keys = list(rp_final.keys())
#half = len(keys) // 2
regionprops1= defaultdict()
for i in rp_final:
    for j in range(len(rp_final[i])):
        regionprops1[i]= [rp_final[i][j][0].centroid, rp_final[i][j][0].area, rp_final[i][j][0].eccentricity, rp_final[i][j][0].orientation,  rp_final[i][j][0].axis_major_length, rp_final[i][j][0].axis_minor_length]
        
# First half of the dictionary
#data_part1 = {key: rp_final[key] for key in keys[:half]}

# Second half of the dictionary
#data_part2 = {key: rp_final[key] for key in keys[half:]}
#with open('/lre/home/gquaglia/background_cassinifull/cosmics4.pkl', 'wb') as f:
 #   pickle.dump(data_part1, f)
with open('/lre/home/gquaglia/background_cassinifull/cosmics_5RS.pkl', 'wb') as f:
    pickle.dump(regionprops1, f)

#with open('/lre/home/gquaglia/background_cassinifull/cosmics1.pkl', 'wb') as f:
    #pickle.dump(rp_final, f)
#
#with open('/lre/home/gquaglia/background_cassinifull/stars1.pkl', 'wb') as f:
   # pickle.dump(stars, f)

#with open('/lre/home/gquaglia/background_cassinifull/satellites1.pkl', 'wb') as f:
 #   pickle.dump(sat, f)
        
#with open('saved_dictionary.pkl', 'rb') as f:
    #loaded_dict = pickle.load(f)