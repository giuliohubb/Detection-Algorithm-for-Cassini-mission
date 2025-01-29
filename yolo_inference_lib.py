import numpy as np
import os
import spiceypy as spice
import scipy as sp
from scipy import ndimage, spatial, signal, stats, optimize
import matplotlib.pyplot as plt
import vicar
import skimage as skim
from skimage import morphology, feature, measure, filters, segmentation, transform
import glob
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
import scipy
from photutils.detection import DAOStarFinder
import higra as hg
import re

def load_images(path, c):
    # Load image (need vicar in the same folder)
    allimgs = sorted(glob.glob(path))
    RER = vicar.VicarImage 
    error_files=[]
    images=defaultdict()
    for i in range(len(allimgs)):
        try:
           
            images[allimgs[i][c:]] = (RER.from_file(allimgs[i]).get_2d_array())
            
        except Exception:
            error_files.append(allimgs[i][c:])
            continue

    return images

def exposure_duration(path, images):
    error= []
    for file in os.listdir(path):
        try:
            for line in open(path + file, "br"):
                if b'EXPOSURE_DURATION' in line:
                    a = str(line)
                    # remove trailing newline
                    if b"FILTER_NAME" in line:
                        b = a[str(line).find('EXPOSURE_DURATION')+18:str(line).find('FILTER_NAME')-1]
                    else:
                        b = a[str(line).find('EXPOSURE_DURATION')+18:str(line).find('GAIN_MODE')-1]
                        
            if float(b) > 1200.0:
                try:
                    del images[file]
                    
                except Exception:
                    error.append(file)
                    continue
        except Exception:
            error.append(file)
            continue

def filter_8bit_images(images):
    
    images8 = defaultdict()
    counter = 0
    idx = []
    for j,i in enumerate(images):
        if images[i].dtype=='uint8' and len(images[i])== 1024:
            images8[i]=images[i]
            counter += 1
        else:
            print(images[i].dtype, len(images[i]))
            idx.insert(0,j)
    
    return images8

def image_division(images8):
    
    size = 512
    overlap = 20
    images8_dict=defaultdict()
    for i, j in enumerate(images8):
        img_number = j
        images8_dict[j] = [images8[img_number[:-4]+'.IMG'][:size+overlap,:size+overlap], images8[img_number[:-4]+'.IMG'][:size+overlap,size-overlap:], images8[img_number[:-4]+'.IMG'][size-overlap:,:size+overlap], images8[img_number[:-4]+'.IMG'][size-overlap:,size-overlap:]]
    
    images8_list = defaultdict()
    for i in images8_dict:
        images8_list[i[:-4]+'.1']=images8_dict[i][0]
        images8_list[i[:-4]+'.2']=images8_dict[i][1]
        images8_list[i[:-4]+'.3']=images8_dict[i][2]
        images8_list[i[:-4]+'.4']=images8_dict[i][3]

    return images8_dict, images8_list

