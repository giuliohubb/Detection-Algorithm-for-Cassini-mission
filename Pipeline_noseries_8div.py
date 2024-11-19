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

mat = flp.import_qmpf('/lre/home/gquaglia/qmpf/')
images = flp.import_images('/lre/home/gquaglia/MyISSdir/*.IMG')
flp.exposure_check(mat, images)
images8, mat = flp.filter_8bit_images(images, mat)
distance_qmpf = flp.distance_from_saturn(mat)
import sys
sys.path.append("/lre/home/gquaglia/notebooks/arago")
import arago
import spiceypy as spice
metakernel = "/lre/home/gquaglia/notebooks/kernels.ker"
spice.furnsh(metakernel)
stars, satellites, corrupted_qmpf = flp.arago_ephemerides('/lre/home/gquaglia/MyISSdir/', '/lre/home/gquaglia/qmpf/', mat)

stars = fs.remove_points_outsideimg(stars)
satellites = fs.remove_points_outsideimg(satellites)
images8_dict, images8_list = flp.images_division(mat, images8, size = 512, overlap = 20, div = 8)
new_sat, new_star = flp.sat_stars_repositioning(images8_dict, stars, satellites, div = 8)

#print(counter_arago, len(corrupted_qmpf))

area_thresh_min = 1
area_thresh_max = 1000
area_thresh = [area_thresh_min,area_thresh_max]
pfa = 0.1
#, corrupted_index, counter_im
detect_clean_im, alt_thresh_im, corrupted_index, counter_im = flp.detect_candidate_cosmic(images8_list, new_star, images8, area_thresh=area_thresh,pfa_bck=pfa,verbose=True, filter_corona=True)

for i in detect_clean_im:
    np.savez('/lre/home/gquaglia/notebooks/4_division/detect_clean/'+str(i), detect_clean_im[i])

with open("/lre/home/gquaglia/notebooks/4_division/corrupted_index.txt", "w") as output:
    for s in corrupted_index:
        output.write(str(s) +"\n")

with open("/lre/home/gquaglia/notebooks/4_division/corrupted_qmpf.txt", "w") as f:
    for s in corrupted_qmpf:
        f.write(str(s) +"\n")
print(len(stars), len(satellites))
for star in stars:
     np.savez('/lre/home/gquaglia/notebooks/4_division/stars/'+str(star), x = stars[star])

for star in satellites:
    print(star)
    np.savez('/lre/home/gquaglia/notebooks/4_division/satellites/'+str(star), x = satellites[star][0], y = satellites[star][1], z = satellites[star][2],w = satellites[star][3], k = satellites[star][4])