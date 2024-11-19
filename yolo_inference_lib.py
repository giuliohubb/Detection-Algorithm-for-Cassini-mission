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
#import deepCR
#import astroscrappy as ac
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
            #print(i, allimgs[i][28:])
            images[allimgs[i][c:]] = (RER.from_file(allimgs[i]).get_2d_array())
            #print('Number of IMAGES file in the directory: ',len(allimgs))  
        except Exception:
            error_files.append(allimgs[i][c:])
            continue

    return images

def distance(path, name, images8):
    
    import sys
    sys.path.append("/lre/home/gquaglia/notebooks/arago")
    import arago
    import spiceypy as spice
    metakernel = "/lre/home/gquaglia/notebooks/kernels.ker"
    spice.furnsh(metakernel)

    et = defaultdict(list)
    distance_qmpf= defaultdict(list)
    altitude_qmpf = defaultdict(list)
    distance_enceladus = defaultdict(list)
    time = defaultdict(list)
    error = []
    for file in os.listdir(path):
        if file in images8.keys():
            #print('ok')
            try:
                for line in open(path +'/'+ file, "br"):
                    if b'STOP_TIME' in line:
                        a = str(line)
                        # remove trailing newline
                        if b"TARGET_DESC" in line:
                            b = a[str(line).find('STOP_TIME')+11:str(line).find('TARGET_DESC')-4]
                            #print(file)
                            if 'Z' in b:
                                #print(file)
                                b = b[1:-1]
                        else:
                            #print('ok', a)
                            b = a[str(line).find('STOP_TIME')+11:str(line).find('TARGET_NAME')-4]
                            if len(b) >= 22:
                                b = b[1:]
                            if 'Z' in b:
                                #print(file)
                                b = b[:-1]
                    #print(b, file)
                time[file].append(b)
                et[file].append(spice.str2et(b))
            except Exception:
                del images8[file]
                error.append(file)
                #print("yes")
                continue             
        else:
            continue
            
    xyz = defaultdict(list)
    xyz_IAU = defaultdict(list)
    xyz_ence = defaultdict(list)
    
    for i in et:
        xyz[i].append(spice.spkpos('CASSINI', et[i], 'J2000', 'NONE', 'Saturn')) 
        xyz_IAU[i].append(spice.spkpos('CASSINI', et[i], 'IAU_SATURN', 'NONE', 'Saturn'))
        xyz_ence[i].append(spice.spkpos('CASSINI', et[i], 'IAU_SATURN', 'NONE', name))
    for i in xyz:
        print(i, time[i], et[i])
        distance_qmpf[i]= np.sqrt(xyz[i][0][0][0][0]**2 + xyz[i][0][0][0][1]**2 + xyz[i][0][0][0][2]**2)/60268 #60268km = equatorial radius of Saturn
        distance_enceladus[i]= np.sqrt(xyz_ence[i][0][0][0][0]**2 + xyz_ence[i][0][0][0][1]**2 + xyz_ence[i][0][0][0][2]**2)/60268
        altitude_qmpf[i] = np.arcsin(xyz_IAU[i][0][0][0][2]/np.sqrt(xyz_IAU[i][0][0][0][0]**2 + xyz_IAU[i][0][0][0][1]**2 + xyz_IAU[i][0][0][0][2]**2)) #latitude with respect to the equator of Saturn: DistancefromSaturn*sin(arctan(y/x))
    
    return distance_qmpf, time, altitude_qmpf, xyz_IAU, distance_enceladus, xyz_ence


def exposure_duration(path, images):
    error= []
    for file in os.listdir(path):
        try:
            for line in open(path +'/'+ file, "br"):
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

def upload_inferred_results(path):

    infer = defaultdict()
    for file in os.listdir(path):
        #print(file)
        with open(path +'/'+file) as f:
            infer[file[:-3]+'QMPF'] = f.readlines()

    yolo_results = defaultdict(list)
    for i in infer:
        for j in range(len(infer[i])):
            #print(j)
            yolo_results[i].append([float(val) for val in infer[i][j].split()])

    return yolo_results

def solve(bl, tr, p) :
   if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
      return True
   else :
      return False
       
def inference_to_regionprops(yolo_results, images8_list, thresh):
    
    import os
    rp_final = defaultdict(list)
    stars = defaultdict(list) 
    sat = defaultdict(list) 
    energy = defaultdict(list)
    gain_dic = defaultdict()
    for file in os.listdir('/lre/home/gquaglia/cassini_full'):      #'/lre/home/gquaglia/infernet'
        for line in open('/lre/home/gquaglia/cassini_full/' + file, "br"):
            #if file == 'N1327295161_1.IMG':
            if b'GAIN_MODE_ID' in line:
                a = str(line)
                #print(a, str(line).find('ELECTRONS PER DN'))
                # remove trailing newline
                if str(line).find('ELECTRONS PER DN') != -1:
                    gain_dic[file] = a[str(line).find('ELECTRONS PER DN')-3:str(line).find('ELECTRONS PER DN')-1]
                else:
                    gain_dic[file] = a[str(line).find('GAIN_MODE_ID')+15:str(line).find('GAIN_MODE_ID')+17]
                   
    for i in sorted(yolo_results):
        print(i, i[:-5])
        print(len(images8_list), len(yolo_results), len(thresh))
        for j in range(len(yolo_results[i])):
            if i[:-5] in images8_list.keys():
                center_x = round(yolo_results[i][j][1]*532)
                center_y = round(yolo_results[i][j][2]*532)
                width = round(yolo_results[i][j][3]*532)
                height = round(yolo_results[i][j][4]*532)
                x_min = max(0,round(center_x - width//2)-10)
                x_max = min(round(center_x + width//2)+10, images8_list[i[:-5]].shape[0])
                y_min = max(0,round(center_y - height//2)-10)
                y_max = min(round(center_y + height//2)+10, images8_list[i[:-5]].shape[1])+10
                
                if i[:-5] in thresh.keys():
                    img_detect_bw = images8_list[i[:-5]][y_min:y_max,x_min:x_max] > thresh[i[:-5]]
                    img_lbl = skim.measure.label(img_detect_bw)
    
                    
                    if yolo_results[i][j][0] == 0:
            
                        a = skim.measure.regionprops(img_lbl)
                        if len(a)>1:
                            for k in range(len(a)):
                                bottom_left = (x_min, y_min)
                                top_right = (x_max, y_max)
                                point = (round(a[k].centroid[0])+x_min, round(a[k].centroid[1])+y_min)
                                if solve(bottom_left, top_right, point)== True:
                                    detected_img = img_lbl == (k+1)
                                    back_mean = np.median(images8_list[i[:-5]][y_min:y_max,x_min:x_max][detected_img==0])
                                    signal_intensity = images8_list[i[:-5]][y_min:y_max,x_min:x_max][detected_img>0].sum()-(back_mean*(len(images8_list[i[:-5]][y_min:y_max,x_min:x_max][detected_img>0])))
                                    energy[i].append(((signal_intensity*int(gain_dic[i[:-6]+'IMG']))*100/35)*(786.5/sp.constants.N_A)*(6.2415E+15))
                                    rp_final[i].append([a[k]])
                                    
                                else:
                                    continue
                        elif len(a)== 0:
                            continue                        
                        else:
                            rp_final[i].append(a)
                            detected_img = img_lbl > 0
                            back_mean = np.median(images8_list[i[:-5]][y_min:y_max,x_min:x_max][detected_img==0])
                            signal_intensity = images8_list[i[:-5]][y_min:y_max,x_min:x_max][detected_img>0].sum()-(back_mean*(len(images8_list[i[:-5]][y_min:y_max,x_min:x_max][detected_img>0])))
                            print(i, gain_dic[i[:-6]+'IMG'])
                            energy[i].append(((signal_intensity*int(gain_dic[i[:-6]+'IMG']))*100/35)*(786.5/sp.constants.N_A)*(6.2415E+15))
                    if yolo_results[i][j][0] == 1:
                        
                        b = skim.measure.regionprops(img_lbl)
                        if len(b)>1:
                            for k in range(len(b)):
                                bottom_left = (x_min, y_min)
                                top_right = (x_max, y_max)
                                point = (round(b[k].centroid[0])+x_min, round(b[k].centroid[1])+y_min)
                                if solve(bottom_left, top_right, point) == True:                                
                                    stars[i].append([b[k]])
                                else:
                                    continue
                        elif len(b)== 0:
                            continue                        
                        else:
                            stars[i].append(b)
                            
                    if yolo_results[i][j][0] == 2:
                        c = skim.measure.regionprops(img_lbl)
                        if len(c)>1:
                            for k in range(len(c)):
                                bottom_left = (x_min, y_min)
                                top_right = (x_max, y_max)
                                point = (round(c[k].centroid[0])+x_min, round(c[k].centroid[1])+y_min)
                                if solve(bottom_left, top_right, point)== True:                                 
                                    sat[i].append([c[k]])
                                else:
                                    continue
                        elif len(c)== 0:
                            continue                        
                        else:
                            sat[i].append(c)

    return rp_final, stars, sat, gain_dic, energy
    
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
            print('vmax (1st percentile): ', vmax)
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
        print('new vmax and new img[bck_mask]: ', vmax, img[bck_mask])
    
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

def build_dataframe(xyz, rp_final, latitude, distance_qmpf, time, images8_dict, new_stars, energy, distance_enceladus, images8_list, thresh):
    
    x = defaultdict()
    y = defaultdict()
    z = defaultdict()
    for i in xyz:
        x[i]= xyz[i][0][0][0][0]
        y[i] = xyz[i][0][0][0][1]
        z[i] = xyz[i][0][0][0][2]
    
    mean_energy=defaultdict()
    for i in energy:
        mean_energy[i] = np.mean(energy[i])
    
    percentage_above_threshold = defaultdict()
    opened_image = defaultdict()
    pixels_below_threshold = defaultdict()
    structuring_element = morphology.disk(3)
    for i in images8_list:
        if i in thresh.keys():
            opened_image[i] = morphology.opening(images8_list[i], structuring_element)        
            below_thresh = opened_image[i]<=thresh[i]+15
            total_pixels = images8_list[i].size
            pixels_below_threshold[i] = np.sum(below_thresh)
        
    
    import datetime as dt
    date_format = '%Y-%jT%H:%M:%S'
    distance_time_cosmics = defaultdict(list)
    for j, i in enumerate(rp_final):
        if rp_final[i] != []:
            #print(i)
            distance_time_cosmics[i]= [dt.datetime.strptime(time[i[:-6]+'IMG'][0][:-4], date_format), distance_qmpf[i[:-6]+'IMG'], round(len(rp_final[i])*(total_pixels/pixels_below_threshold[i[:-5]]),2), latitude[i[:-6]+'IMG'], x[i[:-6]+'IMG'], y[i[:-6]+'IMG'], z[i[:-6]+'IMG'], mean_energy[i], distance_enceladus[i[:-6]+'IMG']]
        
    df_histo=pd.DataFrame.from_dict(distance_time_cosmics, orient='index', columns = ['time','distance','cosmics','latitude', 'x','y','z','energy', 'dist_enc'])
    
    df_img = defaultdict(list)
    for i, j in enumerate(images8_dict):
        #print(j)
        if (j[:-3]+'1.QMPF' in df_histo.index) and (j[:-3]+'2.QMPF' in df_histo.index) and (j[:-3]+'3.QMPF' in df_histo.index) and (j[:-3]+'4.QMPF' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-3]+'1.QMPF'], df_histo['distance'][j[:-3]+'1.QMPF'], df_histo['cosmics'][j[:-3]+'1.QMPF']+df_histo['cosmics'][j[:-3]+'2.QMPF']+df_histo['cosmics'][j[:-3]+'3.QMPF']+df_histo['cosmics'][j[:-3]+'4.QMPF'],df_histo['latitude'][j[:-3]+'1.QMPF'],df_histo['x'][j[:-3]+'1.QMPF'],df_histo['y'][j[:-3]+'1.QMPF'],df_histo['z'][j[:-3]+'1.QMPF'],df_histo['energy'][j[:-3]+'1.QMPF'], df_histo['dist_enc'][j[:-3]+'1.QMPF']]
        elif (j[:-3]+'2.QMPF' in df_histo.index) and (j[:-3]+'3.QMPF' in df_histo.index) and (j[:-3]+'4.QMPF' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-3]+'2.QMPF'], df_histo['distance'][j[:-3]+'2.QMPF'], df_histo['cosmics'][j[:-3]+'2.QMPF']+df_histo['cosmics'][j[:-3]+'3.QMPF']+df_histo['cosmics'][j[:-3]+'4.QMPF'],df_histo['latitude'][j[:-3]+'2.QMPF'], df_histo['x'][j[:-3]+'2.QMPF'],df_histo['y'][j[:-3]+'2.QMPF'],df_histo['z'][j[:-3]+'2.QMPF'],df_histo['energy'][j[:-3]+'2.QMPF'], df_histo['dist_enc'][j[:-3]+'2.QMPF']]
        elif (j[:-3]+'1.QMPF' in df_histo.index) and (j[:-3]+'3.QMPF' in df_histo.index) and (j[:-3]+'4.QMPF' in df_histo.index):   #print(j,i)
            df_img[j]= [df_histo['time'][j[:-3]+'1.QMPF'], df_histo['distance'][j[:-3]+'1.QMPF'], df_histo['cosmics'][j[:-3]+'1.QMPF']+df_histo['cosmics'][j[:-3]+'3.QMPF']+df_histo['cosmics'][j[:-3]+'4.QMPF'],df_histo['latitude'][j[:-3]+'3.QMPF'], df_histo['x'][j[:-3]+'3.QMPF'],df_histo['y'][j[:-3]+'3.QMPF'],df_histo['z'][j[:-3]+'3.QMPF'], df_histo['energy'][j[:-3]+'3.QMPF'],df_histo['dist_enc'][j[:-3]+'3.QMPF']]
        elif (j[:-3]+'2.QMPF' in df_histo.index) and (j[:-3]+'1.QMPF' in df_histo.index) and (j[:-3]+'4.QMPF' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-3]+'1.QMPF'], df_histo['distance'][j[:-3]+'1.QMPF'], df_histo['cosmics'][j[:-3]+'1.QMPF']+df_histo['cosmics'][j[:-3]+'2.QMPF']+df_histo['cosmics'][j[:-3]+'4.QMPF'],df_histo['latitude'][j[:-3]+'4.QMPF'], df_histo['x'][j[:-3]+'4.QMPF'],df_histo['y'][j[:-3]+'4.QMPF'],df_histo['z'][j[:-3]+'4.QMPF'], df_histo['energy'][j[:-3]+'4.QMPF'], df_histo['dist_enc'][j[:-3]+'4.QMPF']]
        elif (j[:-3]+'2.QMPF' in df_histo.index) and (j[:-3]+'3.QMPF' in df_histo.index) and (j[:-3]+'1.QMPF' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-3]+'1.QMPF'], df_histo['distance'][j[:-3]+'1.QMPF'], df_histo['cosmics'][j[:-3]+'1.QMPF']+df_histo['cosmics'][j[:-3]+'2.QMPF']+df_histo['cosmics'][j[:-3]+'3.QMPF'],df_histo['latitude'][j[:-3]+'1.QMPF'], df_histo['x'][j[:-3]+'1.QMPF'],df_histo['y'][j[:-3]+'1.QMPF'],df_histo['z'][j[:-3]+'1.QMPF'], df_histo['energy'][j[:-3]+'1.QMPF'], df_histo['dist_enc'][j[:-3]+'1.QMPF']]
        elif (j[:-3]+'1.QMPF' in df_histo.index) and (j[:-3]+'2.QMPF' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-3]+'1.QMPF'], df_histo['distance'][j[:-3]+'1.QMPF'], df_histo['cosmics'][j[:-3]+'1.QMPF']+df_histo['cosmics'][j[:-3]+'2.QMPF'],df_histo['latitude'][j[:-3]+'1.QMPF'],df_histo['x'][j[:-3]+'1.QMPF'],df_histo['y'][j[:-3]+'1.QMPF'],df_histo['z'][j[:-3]+'1.QMPF'], df_histo['energy'][j[:-3]+'1.QMPF'], df_histo['dist_enc'][j[:-3]+'1.QMPF']]
        elif (j[:-3]+'1.QMPF' in df_histo.index) and (j[:-3]+'3.QMPF' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-3]+'1.QMPF'], df_histo['distance'][j[:-3]+'1.QMPF'], df_histo['cosmics'][j[:-3]+'1.QMPF']+df_histo['cosmics'][j[:-3]+'3.QMPF'],df_histo['latitude'][j[:-3]+'1.QMPF'], df_histo['x'][j[:-3]+'1.QMPF'],df_histo['y'][j[:-3]+'1.QMPF'],df_histo['z'][j[:-3]+'1.QMPF'], df_histo['energy'][j[:-3]+'1.QMPF'], df_histo['dist_enc'][j[:-3]+'1.QMPF']]
        elif (j[:-3]+'1.QMPF' in df_histo.index) and (j[:-3]+'4.QMPF' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-3]+'1.QMPF'], df_histo['distance'][j[:-3]+'1.QMPF'], df_histo['cosmics'][j[:-3]+'1.QMPF']+df_histo['cosmics'][j[:-3]+'4.QMPF'],df_histo['latitude'][j[:-3]+'1.QMPF'], df_histo['x'][j[:-3]+'1.QMPF'],df_histo['y'][j[:-3]+'1.QMPF'],df_histo['z'][j[:-3]+'1.QMPF'], df_histo['energy'][j[:-3]+'1.QMPF'], df_histo['dist_enc'][j[:-3]+'1.QMPF']]
        elif (j[:-3]+'2.QMPF' in df_histo.index) and (j[:-3]+'3.QMPF' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-3]+'2.QMPF'], df_histo['distance'][j[:-3]+'2.QMPF'], df_histo['cosmics'][j[:-3]+'3.QMPF']+df_histo['cosmics'][j[:-3]+'2.QMPF'],df_histo['latitude'][j[:-3]+'2.QMPF'], df_histo['x'][j[:-3]+'2.QMPF'],df_histo['y'][j[:-3]+'2.QMPF'],df_histo['z'][j[:-3]+'2.QMPF'], df_histo['energy'][j[:-3]+'2.QMPF'], df_histo['dist_enc'][j[:-3]+'2.QMPF']]
        elif (j[:-3]+'2.QMPF' in df_histo.index) and (j[:-3]+'4.QMPF' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-3]+'2.QMPF'], df_histo['distance'][j[:-3]+'2.QMPF'], df_histo['cosmics'][j[:-3]+'2.QMPF']+df_histo['cosmics'][j[:-3]+'4.QMPF'],df_histo['latitude'][j[:-3]+'2.QMPF'], df_histo['x'][j[:-3]+'2.QMPF'],df_histo['y'][j[:-3]+'2.QMPF'],df_histo['z'][j[:-3]+'2.QMPF'], df_histo['energy'][j[:-3]+'2.QMPF'], df_histo['dist_enc'][j[:-3]+'2.QMPF']]
        elif (j[:-3]+'3.QMPF' in df_histo.index) and (j[:-3]+'4.QMPF' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-3]+'3.QMPF'], df_histo['distance'][j[:-3]+'3.QMPF'], df_histo['cosmics'][j[:-3]+'3.QMPF']+df_histo['cosmics'][j[:-3]+'4.QMPF'],df_histo['latitude'][j[:-3]+'3.QMPF'], df_histo['x'][j[:-3]+'3.QMPF'],df_histo['y'][j[:-3]+'3.QMPF'],df_histo['z'][j[:-3]+'3.QMPF'], df_histo['energy'][j[:-3]+'3.QMPF'], df_histo['dist_enc'][j[:-3]+'3.QMPF']]
        elif (j[:-3]+'1.QMPF' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-3]+'1.QMPF'], df_histo['distance'][j[:-3]+'1.QMPF'], df_histo['cosmics'][j[:-3]+'1.QMPF'],df_histo['latitude'][j[:-3]+'1.QMPF'], df_histo['x'][j[:-3]+'1.QMPF'],df_histo['y'][j[:-3]+'1.QMPF'],df_histo['z'][j[:-3]+'1.QMPF'], df_histo['energy'][j[:-3]+'1.QMPF'], df_histo['dist_enc'][j[:-3]+'1.QMPF']]
        elif (j[:-3]+'2.QMPF' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-3]+'2.QMPF'], df_histo['distance'][j[:-3]+'2.QMPF'], df_histo['cosmics'][j[:-3]+'2.QMPF'],df_histo['latitude'][j[:-3]+'2.QMPF'], df_histo['x'][j[:-3]+'2.QMPF'],df_histo['y'][j[:-3]+'2.QMPF'],df_histo['z'][j[:-3]+'2.QMPF'], df_histo['energy'][j[:-3]+'2.QMPF'], df_histo['dist_enc'][j[:-3]+'2.QMPF']]
        elif (j[:-3]+'3.QMPF' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-3]+'3.QMPF'], df_histo['distance'][j[:-3]+'3.QMPF'], df_histo['cosmics'][j[:-3]+'3.QMPF'],df_histo['latitude'][j[:-3]+'3.QMPF'], df_histo['x'][j[:-3]+'3.QMPF'],df_histo['y'][j[:-3]+'3.QMPF'],df_histo['z'][j[:-3]+'3.QMPF'], df_histo['energy'][j[:-3]+'3.QMPF'], df_histo['dist_enc'][j[:-3]+'3.QMPF']]
        elif (j[:-3]+'4.QMPFf' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-3]+'4.QMPF'], df_histo['distance'][j[:-3]+'4.QMPF'], df_histo['cosmics'][j[:-3]+'4.QMPF'],df_histo['latitude'][j[:-3]+'4.QMPF'], df_histo['x'][j[:-3]+'4.QMPF'],df_histo['y'][j[:-3]+'4.QMPF'],df_histo['z'][j[:-3]+'4.QMPF'],df_histo['energy'][j[:-3]+'4.QMPF'], df_histo['dist_enc'][j[:-3]+'4.QMPF']]
            #print(j)
    df_final = pd.DataFrame.from_dict(df_img, orient='index', columns = ['time','distance','cosmics','latitude','x','y','z', 'energy', 'dist_enc'])
    
    df_time = df_final.drop(labels = ['distance','latitude'], axis=1)
    df_dist = df_final.drop(labels = ['time','latitude'], axis=1)
    df_lat = df_final.drop(labels = ['time','distance'], axis=1)

    return df_final, df_dist, df_time, df_lat