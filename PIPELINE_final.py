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
mat = flp.import_qmpf('/lre/home/gquaglia/qmpf/')
images = flp.import_images('/lre/home/gquaglia/MyISSdir/*.IMG')
flp.exposure_check(mat, images)
images8, mat = flp.filter_8bit_images(images, mat)
corrupted_qmpf = []
f = open('/lre/home/gquaglia/notebooks/4_division/corrupted_qmpf.txt','r')
for line in f:
    corrupted_qmpf.append(line.strip())
corrupted_index = []
f = open('/lre/home/gquaglia/notebooks/4_division/corrupted_index.txt','r')
for line in f:
    corrupted_index.append(line.strip())
images8_dict, images8_list=flp.images_division(mat, images8, size = 512, overlap = 20, div = 8)
stars = defaultdict(list)
for i in images8_dict:
    #print(i)
    if os.path.exists('/lre/home/gquaglia/notebooks/4_division/stars/' + i+ '.npz'):
        with np.load('/lre/home/gquaglia/notebooks/4_division/stars/' + i+ '.npz') as arc:
            stars[i]= arc['x']
satellites = defaultdict(list)
for i in images8_dict:
    if os.path.exists('/lre/home/gquaglia/notebooks/4_division/satellites/' + i+ '.npz'):
        with np.load('/lre/home/gquaglia/notebooks/4_division/satellites/' + i+ '.npz') as arc:    
            satellites[i]= arc['x'], arc['y'], arc['z'], arc['w'],arc['k']
detected_images = defaultdict(list)
for i in images8_list:
    if os.path.exists('/lre/home/gquaglia/notebooks/4_division/detect_clean/' + i+ '.npz'):
        with np.load('/lre/home/gquaglia/notebooks/4_division/detect_clean/' + i+ '.npz') as arc:
            detected_images[i]= arc['arr_0']
new_sat, new_stars = flp.sat_stars_repositioning(images8_dict, stars, satellites, div = 8)
for i in corrupted_index:
    del new_sat[i]
    del new_stars[i]
    del images8_list[i]
rp_final = flp.regionprops(detected_images)
flp.eliminate_point_on_sat(new_sat, rp_final)
peaks = flp.rp_to_peaks(rp_final)
common_star = flp.common_stars(new_stars, peaks, rp_final)
common_sat = flp.common_sat(new_sat, peaks, rp_final)
for i in corrupted_qmpf:
    #print(i[:-5]+'.1')
    if rp_final[i[:-5]+'.1']:
        del rp_final[i[:-5]+'.1']
    if rp_final[i[:-5]+'.2']:
        del rp_final[i[:-5]+'.2']
    if rp_final[i[:-5]+'.3']:
        del rp_final[i[:-5]+'.3']
    if rp_final[i[:-5]+'.4']:
        del rp_final[i[:-5]+'.4']
    if images8_list[i[:-5]+'.1'].any():
        del images8_list[i[:-5]+'.1']
    if images8_list[i[:-5]+'.2'].any():
        del images8_list[i[:-5]+'.2']
    if images8_list[i[:-5]+'.3'].any():
        del images8_list[i[:-5]+'.3']
    if images8_list[i[:-5]+'.4'].any():
        del images8_list[i[:-5]+'.4']
import csv  

#header = ['x', 'y', 'width', 'height', 'class']

for j,k in enumerate(images8_list):

    with open('/lre/home/gquaglia/notebooks/full_training/labels/' + str(k) + '.txt', 'w', encoding='UTF8') as f:
        #writer = txt.writer(f)

        # write the header
        #writer.writerow(header)
        for i in range(len(rp_final[k][0])):
        # write the data
            #prova_cos = rotate([512, 512], [sure_cosmic[j][i].centroid[1]-diff[j][0], sure_cosmic[j+1][i].centroid[0]-diff[j][1]], (thetas[j]*np.pi)/180)
            f.write(str(0)+' '+ str((rp_final[k][0][i].centroid[1])/532)+' '+ str((rp_final[k][0][i].centroid[0])/532)+' '+ str(((rp_final[k][0][i].bbox[3]-rp_final[k][0][i].bbox[1])+5)/532)+' '+ str(((rp_final[k][0][i].bbox[2]-rp_final[k][0][i].bbox[0])+5)/532))
            f.write('\n')
        
        for i in range(len(common_star[k])):
        # write the data
            f.write(str(1)+' '+ str((common_star[k][i].centroid[1])/532)+' '+ str((common_star[k][i].centroid[0])/532)+' '+ str(((common_star[k][i].bbox[3]-common_star[k][i].bbox[1])+5)/532)+' '+ str(((common_star[k][i].bbox[2]-common_star[k][i].bbox[0])+5)/532))
            f.write('\n')
        for i in range(len(common_sat[k])):
        # write the data            
            f.write(str(2)+' '+ str((common_sat[k][i].centroid[1])/532)+' '+ str((common_sat[k][i].centroid[0])/532)+' '+ str(((common_sat[k][i].bbox[3]-common_sat[k][i].bbox[1])+5)/532)+' '+ str(((common_sat[k][i].bbox[2]-common_sat[k][i].bbox[0])+5)/532))
            f.write('\n')
