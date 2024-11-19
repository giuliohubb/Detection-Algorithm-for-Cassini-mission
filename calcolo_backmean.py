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

images = yil.load_images('/lre/home/gquaglia/cassini_full/*.IMG', 32)
yil.exposure_duration('/lre/home/gquaglia/cassini_full', images)
images8 = yil.filter_8bit_images(images)
images8_dict, images8_list = yil.image_division(images8)
yolo_results = yil.upload_inferred_results('/lre/home/gquaglia/env/yolov5/runs/detect/exp9_cassini_full/labels')

def adaptive_parameter_tuning(img, percentile):
   
    vmax = int(np.percentile(img, percentile))
    return vmax

def estimate_bck_stats(img, img_complete, pfa=0.001,debug=False):
    # Estimate background statistics of given image by fitting a lognormal distribution
    # Return mean, std, and threshold to guarantee a given probability of false alarm (0.01 by default)
    # Set debug = True to have some display of the background estimation process
    #img_number = index//4
    if img.dtype == 'uint8':
        #print(adaptive_parameter_tuning(img, 90))
        if adaptive_parameter_tuning(img, 80)<255:
            if img_complete.min()==0:
                img_nozero = img_complete[img_complete>0]
                vmax = img_nozero.min()+18
            else:
                img_nozero = img_complete[img_complete>0]
                vmax = img_nozero.min()+15
             #   print('Adaptive parameter tuning (80th percentile): ', adaptive_parameter_tuning(img, 80))
             #   print('vmax: ', vmax)
        else:
            adaptive_percentile = 0.5 #15
            vmax = adaptive_parameter_tuning(img, adaptive_percentile)
           # print('vmax (1st percentile): ', vmax)
            #adaptive_percentile = 15
            #vmax = adaptive_parameter_tuning(img, adaptive_percentile)
            # if image is in uint8 format, background values are assumed to range between min(img) and min(img)+30
            #vmax = images8[img_number].min()+15  #30
    else:
        # Else, background values are assumed to range between min(img) and min(img)+100
        vmax = img_complete.min()+100  #100
    bck_mask = (img>0)&(img<=vmax)
    if bck_mask.any()== False:
        vmax = 70
        bck_mask = (img>0)&(img<=vmax)
        #print('new vmax and new img[bck_mask]: ', vmax, img[bck_mask])
    
    # histogram of crude background
    histval,histbins = np.histogram(img[bck_mask],bins=range(img_complete.min(),vmax+1),density=True)
    # assimilate this histogram as a discrete random variable distrbution
    RV = sp.stats.rv_histogram((histval,histbins))
    # sample img_size/10 values out of this random variable distribution
    RV_samples = RV.rvs(size=int(img[bck_mask].size/10))
    # fit a lognormal distribution to previously generated random values
    s,loc,scale = sp.stats.lognorm.fit(RV_samples)
   
    # estimate the mean of the lognormal distribution
    mean_est = sp.stats.lognorm.mean(s,loc=loc,scale=scale)
    # estimate the standard deviation of the lognormal distribution
    std_est = sp.stats.lognorm.std(s,loc=loc,scale=scale)
    # estimate the threshold of the lognormal distribution to get given pfa
    thresh = sp.stats.lognorm.isf(pfa,s,loc=loc,scale=scale)
    
    if debug:
        # display crude background mask
        plt.figure(figsize=(7,5))
        plt.imshow(bck_mask,cmap='gray')
        plt.show()
        print('Crude background mean estimate: %1.3f'%img[bck_mask].mean())
        print('Crude background std estimate: %1.3f'%img[bck_mask].std())
        
        # display histogram + fitted lognormal distribution
        plt.figure(figsize=(7,5))
        histcbins = (histbins+0.5)[:-1]
        plt.plot(histcbins,histval,'b')
        plt.plot(histcbins,sp.stats.lognorm.pdf(histcbins,s,loc=loc,scale=scale),'r')
        plt.hist(RV_samples,bins=range(img_complete.min(),vmax+1),density=True)
        plt.axvline(x=thresh,ymax=0.8)
        plt.show()

        # display estimated background + statistics
        print('Background mean estimate: %1.3f'%mean_est)
        print('Background std estimate: %1.3f'%std_est)
        print('Threshold for %1.3f pfa: %1.3f'%(pfa,thresh))
        plt.figure(figsize=(7,5))
        plt.imshow(img<thresh)
        plt.show()
    return mean_est,std_est,thresh
mean_est = defaultdict()
std_est = defaultdict()
thresh = defaultdict()
corrupted_index = []
counter =0
for i, j in enumerate(images8_list):
    try:
        mean_est[j],std_est[j],thresh[j] = estimate_bck_stats(images8_list[j], images8[j[:-1]+'IMG'], pfa=0.1,debug=False)

    except Exception:
        corrupted_index.append(j)
        counter +=1


with open("/lre/home/gquaglia/background_cassinifull/corrupted_index.txt", "w") as output:
    for s in corrupted_index:
        output.write(str(s) +"\n")

import json

# Specifica il percorso e il nome del file
file_path = '/lre/home/gquaglia/background_cassinifull/dic_mean_nobacknorm.json'

# Salva il dizionario nel file JSON
with open(file_path, 'w') as json_file:
    json.dump(mean_est, json_file, indent=4)  

# Specifica il percorso e il nome del file
file_path1 = '/lre/home/gquaglia/background_cassinifull/dic_std_nobacknorm.json'

# Salva il dizionario nel file JSON
with open(file_path1, 'w') as json_file:
    json.dump(std_est, json_file, indent=4)  

# Specifica il percorso e il nome del file
file_path2 = '/lre/home/gquaglia/background_cassinifull/dic_thresh_nobacknorm.json'

# Salva il dizionario nel file JSON
with open(file_path2, 'w') as json_file:
    json.dump(thresh, json_file, indent=4)  
