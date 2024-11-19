#New library for the image segmentation and with the modified detection algorithm
#Detection algorithm parameter changed: strenght = 2, filter corona = 1.1*alt_thresh (alt_thresh>=20), filter corona = 2.0*alt_thresh (alt_thresh<20), if bck_mask = [] -> vmax=70, if adaptive percentile(80)<255 -> vmax=img.min()+15, else vmax=adaptive percentile(0.5) 
#Image segmentation both 4-Division and 19-Division

from scipy.ndimage import zoom
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

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


def series_from_csv(path_series, path_histo, series_n):
    df = pd.read_csv(path_series, header=None, index_col=0, squeeze = True)
    series = df.to_dict() 
    df_histo = pd.read_csv(path_histo, header=None, squeeze = True)
    df_histo.drop(index = 0)
    files = []
    for i in range(int(df_histo[series_n+1])):
        files.append(series[series_n][2+22*i:20+22*i])
    
    return files

def import_qmpf_from_series(path_qmpf, path_img, files):
    mat = defaultdict(list)
    # Folder Path
    #path = "/Users/gquaglia/Desktop/arago-env/first series/QMPF/"

    # Change the directory
    os.chdir(path_qmpf)

    # Read text File


    def read_text_file(file_path):
        with open(file_path, 'r') as f:
            mat[file]=f.readlines()
    
    for filess in files:
        print(path_img +filess[:13]+'.IMG')
        if os.path.exists(path_img +filess[:13]+'.IMG'):
    # iterate through all file
            for file in os.listdir():

                # Check whether file is in text format or not
                if file.endswith(str(filess)):
                    file_path = f"{path_qmpf}{file}"


                # call read text file function
                    read_text_file(file_path)
    return mat

def import_qmpf(path):
    
    'Import all the .QMPF file in the designed folder'
    
    mat = defaultdict(list)
    # Folder Path
    #path = "/Users/gquaglia/Desktop/arago-env/first series/QMPF/"

    # Change the directory
    os.chdir(path)

    # Read text File


    def read_text_file(file_path):
        with open(file_path, 'r') as f:
            mat[file]=f.readlines()


    # iterate through all file
    for file in os.listdir():
        
        # Check whether file is in text format or not
        if file.endswith(".QMPF"):
            file_path = f"{path}{file}"

            # call read text file function
            read_text_file(file_path)
    return mat

def distance_from_saturn(mat, name):
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
    QMPF = []
    time = defaultdict(list)
    word = 'STOP_TIME'
    for qmpf in sorted(mat):
        QMPF.append(qmpf)
        for line in mat[qmpf]:
            if line.find(word) != -1:
                a =line
        a1 = a[12:33]
        time[qmpf].append(a1)
        et[qmpf].append(spice.str2et(a1))

    xyz = defaultdict(list)
    xyz_IAU = defaultdict(list)
    xyz_ence = defaultdict(list)
    for i in et:
        xyz[i].append(spice.spkpos('CASSINI', et[i], 'J2000', 'NONE', 'Saturn')) 
        xyz_IAU[i].append(spice.spkpos('CASSINI', et[i], 'IAU_SATURN', 'NONE', 'Saturn'))
        xyz_ence[i].append(spice.spkpos('CASSINI', et[i], 'IAU_SATURN', 'NONE', name))
    for i in xyz:
        distance_qmpf[i]= np.sqrt(xyz[i][0][0][0][0]**2 + xyz[i][0][0][0][1]**2 + xyz[i][0][0][0][2]**2)/60268 #60268km = equatorial radius of Saturn
        distance_enceladus[i]= np.sqrt(xyz_ence[i][0][0][0][0]**2 + xyz_ence[i][0][0][0][1]**2 + xyz_ence[i][0][0][0][2]**2)/60268
        altitude_qmpf[i] = np.arcsin(xyz_IAU[i][0][0][0][2]/np.sqrt(xyz_IAU[i][0][0][0][0]**2 + xyz_IAU[i][0][0][0][1]**2 + xyz_IAU[i][0][0][0][2]**2)) #Altitude with respect to the equator of Saturn: DistancefromSaturn*sin(arctan(y/x))

    return distance_qmpf, time, altitude_qmpf, xyz_IAU, distance_enceladus

def remove_points_outsideimg(stars):
    
    '''Takes the dictionary with all the stars/satellites for every image and return the same dictionary without the points that are
    outside of the image'''
    
    index = defaultdict(list)
    for star in stars:
        for i in range(len(stars[star][0])):
            if stars[star][0][i]<0 or stars[star][0][i]>1024 or stars[star][1][i]<0 or stars[star][1][i]>1024:
                    index[star].insert(0,i)

    for star in stars:
        for i in range(len(index[star])):
            stars[star][0] = np.delete(stars[star][0],index[star][i])
            stars[star][1] = np.delete(stars[star][1],index[star][i])
            
    return stars    

def import_images_from_serie(path_img, files):
    imagez = []
    for file in files:
        images = import_images(path_img + str(file[:-5])+'.IMG')
        #print(images)
        imagez.append(images)
    #print(imagez)
    image = []
    for i in range(len(imagez)):
        image.append(imagez[i][0])
    print('Number of IMAGES file in the directory: ',len(image))  
    return image

def import_images(path):
    # Load image (need vicar in the same folder)
    allimgs = sorted(glob.glob(path))
    RER = vicar.VicarImage 
    
    images=defaultdict()
    for i in range(len(allimgs)):
        images[allimgs[i][28:]] = (RER.from_file(allimgs[i]).get_2d_array())
        #print('Number of IMAGES file in the directory: ',len(allimgs))  
    
    return images

def exposure_check(mat, image): #files):
    
    '''Check if the exposure duration of every image of the series is < 1s'''
    
    word = 'EXPOSURE_DURATION'
    index = []
    #i=0
    for qmpf in sorted(mat):
        for line in mat[qmpf]:
            # check if string present on a current line
            if line.find(word) != -1:
                #print(word, 'string exists in file')
               # print('Line Number:', mat[qmpf].index(line))
               # print('Line:', line)
                
                if float(mat[qmpf][mat[qmpf].index(line)][20:-1])>1000.0:
                    
                    #print('exposure duration of '+ qmpf + ' = ' + str(mat[qmpf][mat[qmpf].index(line)][19:-1]))
                    del mat[qmpf]
                    del image[qmpf[:-4] + 'IMG']
                    #index.append(i)
                #i=i+1
                
    #index.reverse()
    #print(index)
    #for i in range(len(index)):
     #   del image[index[i]]
        #del files[index[i]]
        
def upload_binary_images(path):
    allimgs = glob.glob(path)
    allimgs.sort(key=natural_keys)
    index_img=[]
    for i in allimgs:
        #print(i[51:-4])
        index_img.append(int(i[51:-4]))
    detected_images = defaultdict(list)
    for i in range(len(allimgs)):
        detected_images[index_img[i]].append(np.load(allimgs[i]))
    
    return detected_images  

def eliminate_sat_star_nodetection(mat,new_sat,new_star,corrupted_index):
    nogood_stars=[]
    for i, j in enumerate(sorted(mat)):
            if i*4 in corrupted_index:
                nogood_stars.insert(0,j[:-5]+'.1')
            if (i*4+1) in corrupted_index:
                nogood_stars.insert(0,j[:-5]+'.2')
            if (i*4+2) in corrupted_index:
                nogood_stars.insert(0,j[:-5]+'.3')
            if (i*4+3) in corrupted_index:
                nogood_stars.insert(0,j[:-5]+'.4')
                
    for i in nogood_stars:    
        del new_star[i]
        del new_sat[i]   

def clean_detected_images(mat, corrupted_qmpf, detected_images):
    badqmpf_index=[]
    for i, j in enumerate(sorted(mat)):
        if j in corrupted_qmpf:
            badqmpf_index.append(i)
            
    index_qmpf_tomerge=[]   
    for i in badqmpf_index:
        index_qmpf_tomerge.append(i*4)
        index_qmpf_tomerge.append(i*4+1)
        index_qmpf_tomerge.append(i*4+2)
        index_qmpf_tomerge.append(i*4+3)
        
    bad_qmpf_detected=[]
    for i in index_qmpf_tomerge:
        if i in detected_images:
            bad_qmpf_detected.insert(0,i)
            
    for i in bad_qmpf_detected:
        del detected_images[i]
        
    detected_images_list=[]
    for i in detected_images:
        detected_images_list.append(detected_images[i])
        
    return detected_images_list

def arago_ephemerides(path_img,path_qmpf, mat):
    
    import arago
    from arago import AragoException
    import spiceypy as spice
    metakernel = "/lre/home/gquaglia/notebooks/kernels.ker"
    spice.furnsh(metakernel)

    from arago.processor import Processor
    import arago.qmpf
    stars = defaultdict(list)
    satellites = defaultdict(list)
    QMPF = []
    corrupted_qmpf = []
    counter = 0
    word = 'EXPOSURE_DURATION'
    for qmpf in sorted(mat):
        if os.path.exists(path_img +qmpf[:-5]+'.IMG'):
            for line in mat[qmpf]:
                if line.find(word) != -1:
                    a =line


            #path_img = '/Users/gquaglia/Desktop/arago1/ALLIMAGES_CASSINI/MyISSdir/'
            #path_qmpf = '/Users/gquaglia/Desktop/arago1/ALLIMAGES_CASSINI/qmpf/'
            vicar_img = vicar.VicarImage.from_file(path_img + qmpf[:-5]+'.IMG')
            vicar_img.path = path_img + qmpf[:-5]+'.IMG'
            proc = Processor(vicar_img)
            arago.settings.CATALOG_MAGNITUDE = 0.00171878 * float(a[20:-1]) + 11.8303
            try:
                arago.qmpf.importQMPF(path_qmpf + qmpf, proc, overwrite_catalog=False)
                proc.load_star_catalog()
                proc.load_visible_satellites(target_name=None)
            except Exception:
                corrupted_qmpf.append(qmpf)
                counter += 1
                continue

            QMPF.append(qmpf)   
            stars[qmpf]= [proc.cat_stars.sp, proc.cat_stars.ln]
            satellites[qmpf]= [proc.satellites.sp, proc.satellites.ln,proc.satellites.radii,proc.satellites.res_px,proc.satellites.name]### Import all the IMAGES of the first series taken into account
            
    return stars, satellites, corrupted_qmpf


def filter_8bit_images(images, mat):
    counter = 0
    images8 = defaultdict()
    idx = []
    for j,i in enumerate(images):
        if images[i].dtype=='uint8':
            images8[i]=images[i]
            counter += 1
        else:
            idx.insert(0,j)

    for i in idx:
    #print(i)
        del mat[sorted(mat)[i]]
        
    return images8, mat

def images_division(mat, images8, size = 512, overlap = 20, div = 8):
    

    #size = 512
    #overlap = 20 
    images8_dict=defaultdict()
    for i, j in enumerate(sorted(mat)):
        img_number = j
        images8_dict[j] = [images8[img_number[:-4]+'IMG'][:size+overlap,:size+overlap], images8[img_number[:-4]+'IMG'][:size+overlap,size-overlap:], images8[img_number[:-4]+'IMG'][size-overlap:,:size+overlap], images8[img_number[:-4]+'IMG'][size-overlap:,size-overlap:]]

    images8_list = defaultdict()
    for i in images8_dict:
        images8_list[i[:-5]+'.1']=images8_dict[i][0]
        images8_list[i[:-5]+'.2']=images8_dict[i][1]
        images8_list[i[:-5]+'.3']=images8_dict[i][2]
        images8_list[i[:-5]+'.4']=images8_dict[i][3]
            
    if div==16:
        size = int((size+overlap)/2)
        
        #overlap = 20 
        images16_dict=defaultdict()
        for i, j in enumerate(images8_list):
            img_number = j
            images16_dict[j] = [images8_list[img_number][:size+overlap,:size+overlap], images8_list[img_number][:size+overlap,size-overlap:], images8_list[img_number][size-overlap:,:size+overlap], images8_list[img_number][size-overlap:,size-overlap:]]

        images16_list = defaultdict()
        for i in images8_dict:
            images16_list[i[:-5]+'.1']=images16_dict[i[:-5]+'.1'][0]
            images16_list[i[:-5]+'.2']=images16_dict[i[:-5]+'.1'][1]
            images16_list[i[:-5]+'.3']=images16_dict[i[:-5]+'.2'][0]
            images16_list[i[:-5]+'.4']=images16_dict[i[:-5]+'.2'][1]    

            images16_list[i[:-5]+'.5']=images16_dict[i[:-5]+'.1'][2]
            images16_list[i[:-5]+'.6']=images16_dict[i[:-5]+'.1'][3]
            images16_list[i[:-5]+'.7']=images16_dict[i[:-5]+'.2'][2]
            images16_list[i[:-5]+'.8']=images16_dict[i[:-5]+'.2'][3]

            images16_list[i[:-5]+'.9']=images16_dict[i[:-5]+'.3'][0]
            images16_list[i[:-5]+'.10']=images16_dict[i[:-5]+'.3'][1]
            images16_list[i[:-5]+'.11']=images16_dict[i[:-5]+'.4'][0]
            images16_list[i[:-5]+'.12']=images16_dict[i[:-5]+'.4'][1]    

            images16_list[i[:-5]+'.13']=images16_dict[i[:-5]+'.3'][2]
            images16_list[i[:-5]+'.14']=images16_dict[i[:-5]+'.3'][3]
            images16_list[i[:-5]+'.15']=images16_dict[i[:-5]+'.4'][2]
            images16_list[i[:-5]+'.16']=images16_dict[i[:-5]+'.4'][3]
            
        return images8_dict, images8_list, images16_dict, images16_list
    
    else:
        return images8_dict, images8_list

def eliminate_point_on_sat(new_sat, rp_image):
    idx=[]
    for i, j in enumerate(new_sat):
        for ii in range(len(new_sat[j])):
            a = new_sat[j][ii][2]/new_sat[j][ii][3]
            idx=[]
            if a>5:
                index_todel=[]
                if rp_image[j]!=[]:
                    for jj in range(len(rp_image[j][0])):       
                        if ((rp_image[j][0][jj].centroid[1]-new_sat[j][ii][0])**2+(rp_image[j][0][jj].centroid[0]-new_sat[j][ii][1])**2<(a)**2):
                            index_todel.insert(0,jj)
                            #print(i, index_todel)

                    for k in range(len(index_todel)):
                        rp_image[j][0].pop(index_todel[k])

                    idx.insert(0,ii)
                    #print(i, idx)
        for kk in range(len(idx)):
            if new_sat[j]!= []:
                #print(j,i)
                new_sat[j].pop(idx[kk])
                
def sat_stars_repositioning(images8_dict, stars, satellites, div = 8):
    
    if div ==8:
        #img_size=532
        
        new_star=defaultdict()
        new_sat = defaultdict()

        for k, j in enumerate(images8_dict):
            starprov = stars[j]
            satprov = satellites[j]
            new_st1=[]
            new_st2=[]
            new_st3=[]
            new_st4=[]
            new_sat1=[]
            new_sat2=[]
            new_sat3=[]
            new_sat4=[]
            if len(starprov) != 0:
                for i in range(len(starprov[0])):
                    if (starprov[0][i]<= 532) & (starprov[1][i]<= 532):
                        new_st1.append([starprov[0][i],starprov[1][i]])
                        #new_star['N1455662244_1.1'].append([starprov[0][i],starprov[1][i]])
                    if (starprov[0][i]>= 492) & (starprov[1][i]<= 532):
                        new_st2.append([starprov[0][i]-492, starprov[1][i]])
                        #new_star['N1455662244_1.2'].append([starprov[0][i], starprov[1][i]-502])
                    if (starprov[0][i]<= 532) & (starprov[1][i]>= 492):
                        new_st3.append([starprov[0][i], starprov[1][i]-492])
                        #new_star['N1455662244_1.3'].append([starprov[0][i]-502, starprov[1][i]])
                    if (starprov[1][i]>= 492) & (starprov[0][i]>= 492):
                        new_st4.append([starprov[0][i]-492, starprov[1][i]-492])
                        #new_star['N1455662244_1.4'].append([starprov[0][i]-502, starprov[1][i]-502])
            new_star[j[:-5]+'.1']=new_st1
            new_star[j[:-5]+'.2']=new_st2
            new_star[j[:-5]+'.3']=new_st3
            new_star[j[:-5]+'.4']=new_st4
            if len(satprov) != 0:
                for i in range(len(satprov[0])):
                    if (satprov[0][i]<= 532) & (satprov[1][i]<= 532):
                        new_sat1.append([satprov[0][i],satprov[1][i], np.mean(satprov[2][i]),satprov[3][i]])
                        #new_star['N1455662244_1.1'].append([starprov[0][i],starprov[1][i]])
                    if (satprov[0][i]>= 492) & (satprov[1][i]<= 532):
                        new_sat2.append([satprov[0][i]-492, satprov[1][i], np.mean(satprov[2][i]),satprov[3][i]])
                        #new_star['N1455662244_1.2'].append([starprov[0][i], starprov[1][i]-502])
                    if (satprov[0][i]<= 532) & (satprov[1][i]>= 492):
                        new_sat3.append([satprov[0][i], satprov[1][i]-492, np.mean(satprov[2][i]),satprov[3][i]])
                        #new_star['N1455662244_1.3'].append([starprov[0][i]-502, starprov[1][i]])
                    if (satprov[1][i]>= 492) & (satprov[0][i]>= 492):
                        #print(j,i)
                        #print(len(satprov))
                        #print(satprov)
                        new_sat4.append([satprov[0][i]-492, satprov[1][i]-492, np.mean(satprov[2][i]),satprov[3][i]])
                        #new_star['N1455662244_1.4'].append([starprov[0][i]-502, starprov[1][i]-502])
            new_sat[j[:-5]+'.1']=new_sat1
            new_sat[j[:-5]+'.2']=new_sat2
            new_sat[j[:-5]+'.3']=new_sat3
            new_sat[j[:-5]+'.4']=new_sat4
                            
        return new_sat, new_star  
    
    if div == 16:
        
        new_star16=defaultdict()

        for k, j in enumerate(images8_dict):
            starprov = stars[j]
            #print(starprov)
            satprov = satellites[j]
            new_st1=[]
            new_st2=[]
            new_st3=[]
            new_st4=[]
            new_st5=[]
            new_st6=[]
            new_st7=[]
            new_st8=[]
            new_st9=[]
            new_st10=[]
            new_st11=[]
            new_st12=[]
            new_st13=[]
            new_st14=[]
            new_st15=[]
            new_st16=[]
            if len(starprov) != 0:
                for i in range(len(starprov[0])):
                    if (starprov[0][i]<= 286) & (starprov[1][i]<= 286):
                        new_st1.append([starprov[0][i],starprov[1][i]])
                        #new_star['N1455662244_1.1'].append([starprov[0][i],starprov[1][i]])
                    if (starprov[0][i]>= 246) & (starprov[1][i]<= 286) & (starprov[0][i]<= 532):
                        new_st2.append([starprov[0][i]-246, starprov[1][i]])
                        #new_star['N1455662244_1.2'].append([starprov[0][i], starprov[1][i]-502])
                    if (starprov[0][i]>= 492) & (starprov[1][i]<= 286) & (starprov[0][i]<= 778):
                        new_st3.append([starprov[0][i]-492, starprov[1][i]])
                        #new_star['N1455662244_1.3'].append([starprov[0][i]-502, starprov[1][i]])
                    if (starprov[0][i]>= 738) & (starprov[1][i]<= 286):
                        new_st4.append([starprov[0][i]-738, starprov[1][i]])
                        #new_star['N1455662244_1.4'].append([starprov[0][i]-502, starprov[1][i]-502])

                    if (starprov[0][i]<= 286) & (starprov[1][i]>= 246) & (starprov[1][i]<= 532):
                        new_st5.append([starprov[0][i],starprov[1][i]-246])
                        #new_star['N1455662244_1.1'].append([starprov[0][i],starprov[1][i]])
                    if (starprov[0][i]>= 246) & (starprov[1][i]>= 246) & (starprov[1][i]<= 532) & (starprov[0][i]<= 532):
                        new_st6.append([starprov[0][i]-246, starprov[1][i]-246])
                        #new_star['N1455662244_1.2'].append([starprov[0][i], starprov[1][i]-502])
                    if (starprov[0][i]>= 492) & (starprov[1][i]>= 246) & (starprov[1][i]<= 532) & (starprov[0][i]<= 778):
                        new_st7.append([starprov[0][i]-492, starprov[1][i]-246])
                        #new_star['N1455662244_1.3'].append([starprov[0][i]-502, starprov[1][i]])
                    if (starprov[0][i]>= 738) & (starprov[1][i]>= 246) & (starprov[1][i]<= 532):
                        new_st8.append([starprov[0][i]-738, starprov[1][i]-246])
                        #new_star['N1455662244_1.4'].append([starprov[0][i]-502, starprov[1][i]-502])

                    if (starprov[0][i]<= 286) & (starprov[1][i]>= 492) & (starprov[1][i]<= 778):
                        new_st9.append([starprov[0][i],starprov[1][i]-492])
                        #new_star['N1455662244_1.1'].append([starprov[0][i],starprov[1][i]])
                    if (starprov[0][i]>= 246) & (starprov[1][i]>= 492) & (starprov[1][i]<= 778) & (starprov[0][i]<= 532):
                        new_st10.append([starprov[0][i]-246, starprov[1][i]-492])
                        #new_star['N1455662244_1.2'].append([starprov[0][i], starprov[1][i]-502])
                    if (starprov[0][i]>= 492) & (starprov[1][i]>= 492) & (starprov[1][i]<= 778) & (starprov[0][i]<= 778):
                        new_st11.append([starprov[0][i]-492, starprov[1][i]-492])
                        #new_star['N1455662244_1.3'].append([starprov[0][i]-502, starprov[1][i]])
                    if (starprov[0][i]>= 738) & (starprov[1][i]>= 492) & (starprov[1][i]<= 778):
                        new_st12.append([starprov[0][i]-738, starprov[1][i]-492])
                        #new_star['N1455662244_1.4'].append([starprov[0][i]-502, starprov[1][i]-502])

                    if (starprov[0][i]<= 286) & (starprov[1][i]>= 738):
                        new_st13.append([starprov[0][i],starprov[1][i]-738])
                        #new_star['N1455662244_1.1'].append([starprov[0][i],starprov[1][i]])
                    if (starprov[0][i]>= 246) & (starprov[1][i]>= 738) & (starprov[0][i]<= 532):
                        new_st14.append([starprov[0][i]-246, starprov[1][i]-738])
                        #new_star['N1455662244_1.2'].append([starprov[0][i], starprov[1][i]-502])
                    if (starprov[0][i]>= 492) & (starprov[1][i]>= 738) & (starprov[0][i]<= 778):
                        new_st15.append([starprov[0][i]-492, starprov[1][i]-738])
                        #new_star['N1455662244_1.3'].append([starprov[0][i]-502, starprov[1][i]])
                    if (starprov[0][i]>= 738) & (starprov[1][i]>= 738):
                        new_st16.append([starprov[0][i]-738, starprov[1][i]-738])
                        #new_star['N1455662244_1.4'].append([starprov[0][i]-502, starprov[1][i]-502])


            new_star16[j[:-5]+'.1']=new_st1
            new_star16[j[:-5]+'.2']=new_st2
            new_star16[j[:-5]+'.3']=new_st3
            new_star16[j[:-5]+'.4']=new_st4
            new_star16[j[:-5] + '.5'] = new_st5
            new_star16[j[:-5] + '.6'] = new_st6
            new_star16[j[:-5] + '.7'] = new_st7
            new_star16[j[:-5] + '.8'] = new_st8
            new_star16[j[:-5] + '.9'] = new_st9
            new_star16[j[:-5] + '.10'] = new_st10
            new_star16[j[:-5] + '.11'] = new_st11
            new_star16[j[:-5] + '.12'] = new_st12
            new_star16[j[:-5] + '.13'] = new_st13
            new_star16[j[:-5] + '.14'] = new_st14
            new_star16[j[:-5] + '.15'] = new_st15
            new_star16[j[:-5] + '.16'] = new_st16

        new_satt16 = defaultdict(list)

        for k, j in enumerate(images8_dict):
            #print(satprov)
            satprov = satellites[j]
            new_sat1=[]
            new_sat2=[]
            new_sat3=[]
            new_sat4=[]
            new_sat5=[]
            new_sat6=[]
            new_sat7=[]
            new_sat8=[]
            new_sat9=[]
            new_sat10=[]
            new_sat11=[]
            new_sat12=[]
            new_sat13=[]
            new_sat14=[]
            new_sat15=[]
            new_sat16=[]
            if len(satprov) != 0:
                for i in range(len(satprov[0])):
                    if (satprov[0][i]<= 286) & (satprov[1][i]<= 286):
                        new_sat1.append([satprov[0][i],satprov[1][i], np.mean(satprov[2][i]),satprov[3][i]])
                        #new_satar['N1455662244_1.1'].append([satprov[0][i],satprov[1][i]])
                    if (satprov[0][i]>= 246) & (satprov[1][i]<= 286) & (satprov[0][i]<= 532):
                        new_sat2.append([satprov[0][i]-246, satprov[1][i], np.mean(satprov[2][i]),satprov[3][i]])
                        #new_satar['N1455662244_1.2'].append([satprov[0][i], satprov[1][i]-502])
                    if (satprov[0][i]>= 492) & (satprov[1][i]<= 286) & (satprov[0][i]<= 778):
                        new_sat3.append([satprov[0][i]-492, satprov[1][i], np.mean(satprov[2][i]),satprov[3][i]])
                        #new_satar['N1455662244_1.3'].append([satprov[0][i]-502, satprov[1][i]])
                    if (satprov[0][i]>= 738) & (satprov[1][i]<= 286):
                        new_sat4.append([satprov[0][i]-738, satprov[1][i],np.mean(satprov[2][i]),satprov[3][i]])
                        #new_satar['N1455662244_1.4'].append([satprov[0][i]-502, satprov[1][i]-502])

                    if (satprov[0][i]<= 286) & (satprov[1][i]>= 246) & (satprov[1][i]<= 532):
                        new_sat5.append([satprov[0][i],satprov[1][i]-246,np.mean(satprov[2][i]),satprov[3][i]])
                        #new_satar['N1455662244_1.1'].append([satprov[0][i],satprov[1][i]])
                    if (satprov[0][i]>= 246) & (satprov[1][i]>= 246) & (satprov[1][i]<= 532) & (satprov[0][i]<= 532):
                        new_sat6.append([satprov[0][i]-246, satprov[1][i]-246,np.mean(satprov[2][i]),satprov[3][i]])
                        #new_satar['N1455662244_1.2'].append([satprov[0][i], satprov[1][i]-502])
                    if (satprov[0][i]>= 492) & (satprov[1][i]>= 246) & (satprov[1][i]<= 532) & (satprov[0][i]<= 778):
                        new_sat7.append([satprov[0][i]-492, satprov[1][i]-246,np.mean(satprov[2][i]),satprov[3][i]])
                        #new_satar['N1455662244_1.3'].append([satprov[0][i]-502, satprov[1][i]])
                    if (satprov[0][i]>= 738) & (satprov[1][i]>= 246) & (satprov[1][i]<= 532):
                        new_sat8.append([satprov[0][i]-738, satprov[1][i]-246,np.mean(satprov[2][i]),satprov[3][i]])
                        #new_satar['N1455662244_1.4'].append([satprov[0][i]-502, satprov[1][i]-502])

                    if (satprov[0][i]<= 286) & (satprov[1][i]>= 492) & (satprov[1][i]<= 778):
                        new_sat9.append([satprov[0][i],satprov[1][i]-492,np.mean(satprov[2][i]),satprov[3][i]])
                        #new_satar['N1455662244_1.1'].append([satprov[0][i],satprov[1][i]])
                    if (satprov[0][i]>= 246) & (satprov[1][i]>= 492) & (satprov[1][i]<= 778) & (satprov[0][i]<= 532):
                        new_sat10.append([satprov[0][i]-246, satprov[1][i]-492,np.mean(satprov[2][i]),satprov[3][i]])
                        #new_satar['N1455662244_1.2'].append([satprov[0][i], satprov[1][i]-502])
                    if (satprov[0][i]>= 492) & (satprov[1][i]>= 492) & (satprov[1][i]<= 778) & (satprov[0][i]<= 778):
                        new_sat11.append([satprov[0][i]-492, satprov[1][i]-492,np.mean(satprov[2][i]),satprov[3][i]])
                        #new_satar['N1455662244_1.3'].append([satprov[0][i]-502, satprov[1][i]])
                    if (satprov[0][i]>= 738) & (satprov[1][i]>= 492) & (satprov[1][i]<= 778):
                        new_sat12.append([satprov[0][i]-738, satprov[1][i]-492,np.mean(satprov[2][i]),satprov[3][i]])
                        #new_satar['N1455662244_1.4'].append([satprov[0][i]-502, satprov[1][i]-502])

                    if (satprov[0][i]<= 286) & (satprov[1][i]>= 738):
                        new_sat13.append([satprov[0][i],satprov[1][i]-738,np.mean(satprov[2][i]),satprov[3][i]])
                        #new_satar['N1455662244_1.1'].append([satprov[0][i],satprov[1][i]])
                    if (satprov[0][i]>= 246) & (satprov[1][i]>= 738) & (satprov[0][i]<= 532):
                        new_sat14.append([satprov[0][i]-246, satprov[1][i]-738,np.mean(satprov[2][i]),satprov[3][i]])
                        #new_satar['N1455662244_1.2'].append([satprov[0][i], satprov[1][i]-502])
                    if (satprov[0][i]>= 492) & (satprov[1][i]>= 738) & (satprov[0][i]<= 778):
                        new_sat15.append([satprov[0][i]-492, satprov[1][i]-738,np.mean(satprov[2][i]),satprov[3][i]])
                        #new_satar['N1455662244_1.3'].append([satprov[0][i]-502, satprov[1][i]])
                    if (satprov[0][i]>= 738) & (satprov[1][i]>= 738):
                        new_sat16.append([satprov[0][i]-738, satprov[1][i]-738, np.mean(satprov[2][i]),satprov[3][i]])
                        #new_satar['N1455662244_1.4'].append([satprov[0][i]-502, satprov[1][i]-502])


            new_satt16[j[:-5]+'.1']= new_sat1
            new_satt16[j[:-5]+'.2']= new_sat2
            new_satt16[j[:-5]+'.3']= new_sat3
            new_satt16[j[:-5]+'.4']= new_sat4
            new_satt16[j[:-5] + '.5'] = new_sat5
            new_satt16[j[:-5] + '.6'] = new_sat6
            new_satt16[j[:-5] + '.7'] = new_sat7
            new_satt16[j[:-5] + '.8'] = new_sat8
            new_satt16[j[:-5] + '.9'] = new_sat9
            new_satt16[j[:-5] + '.10'] = new_sat10
            new_satt16[j[:-5] + '.11'] = new_sat11
            new_satt16[j[:-5] + '.12'] = new_sat12
            new_satt16[j[:-5] + '.13'] = new_sat13
            new_satt16[j[:-5] + '.14'] = new_sat14
            new_satt16[j[:-5] + '.15'] = new_sat15
            new_satt16[j[:-5] + '.16'] = new_sat16

        return new_satt16, new_star16        
        
        

def rp_to_peaks(rp_image):
    
    peaks = defaultdict(list)
    for i,j in enumerate(rp_image):
        if rp_image[j] != []:
            for rp in rp_image[j][0]:
                peaks[j].append([rp.centroid[1],rp.centroid[0]])

    return peaks

def common_stars(new_star, peaks, rp_image):
    
    x_star = defaultdict(list)
    y_star = defaultdict(list)
    for i, j in enumerate(new_star):
        if (peaks[j]!=[]) & (new_star[j]!=[]):
            #print(i)
            spatial_distance = sp.spatial.distance.cdist(peaks[j],new_star[j])
            close_pixels = spatial_distance<3
            x_star[j],y_star[j] = np.where(close_pixels)
    rev_star = defaultdict(list)
    for i, j in enumerate(x_star):
        b = sorted([*set(x_star[j].tolist())])
        b.reverse()
        #print(b)
        rev_star[j] = b

    common_star = defaultdict(list)
    #common_all = defaultdict(list)
    for j in peaks:
        for i in range(len(rev_star[j])):
            #print(i, j)
            common_star[j].append(rp_image[j][0][rev_star[j][i]])
            rp_image[j][0].pop(rev_star[j][i])
            peaks[j].pop(rev_star[j][i])
    
    return common_star

def common_sat(new_sat, peaks, rp_image):
    
    new_st = defaultdict(list)
    for i,j in enumerate(new_sat):
        #print(j)
        if new_sat[j]!=[]:
            for ii in range(len(new_sat[j])): 
                #print(j)
                new_st[j].append([new_sat[j][ii][0],new_sat[j][ii][1]])
        else:
            new_st[j]=[]
            
    x_sat = defaultdict(list)
    y_sat = defaultdict(list)
    for i, j in enumerate(new_sat):
        if (peaks[j]!=[]) & (new_st[j]!=[]):
            #print(i)
            spatial_distance = sp.spatial.distance.cdist(peaks[j],new_st[j])
            close_pixels = spatial_distance<5
            x_sat[j],y_sat[j] = np.where(close_pixels)
    rev_sat = defaultdict(list)
    for i,j in enumerate(x_sat):
        b = sorted([*set(x_sat[j].tolist())])
        b.reverse()
        #print(b)
        rev_sat[j] = b

    common_sat = defaultdict(list)
    #common_all = defaultdict(list)
    for j in peaks:
        for i in range(len(rev_sat[j])):
            #print(i, j)
            common_sat[j].append(rp_image[j][0][rev_sat[j][i]])
            rp_image[j][0].pop(rev_sat[j][i])
            
    return common_sat

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
                vmax = img_complete.min()+18
            else:
                vmax = img_complete.min()+15
                print('Adaptive parameter tuning (80th percentile): ', adaptive_parameter_tuning(img, 80))
                print('vmax: ', vmax)
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

def get_corona_region(reg,strel=skim.morphology.square(3)):
    # Extract outer border (corona) of input binary region
    # Structuring element is square by default (8-connexity border), can also be changed to 3x3 cross
    dilreg = skim.morphology.dilation(reg,strel)
    return dilreg ^ reg

def revome_border_CCs(img_lbl,width=4):
    # Remove all labeled connected components that are less than 5 pixels from any image border
    # Return a binary image (/!\ but input is a label image, NOT a binary image)
    r,c = img_lbl.shape
    frame_bw = np.ones((r,c)).astype(bool)
    frame_bw[width:-width,width:-width] = False
    lbl_borders = np.unique(img_lbl[frame_bw])
    for lbl in lbl_borders:
        img_lbl[img_lbl==lbl]=0    
    return img_lbl > 0

def filter_corona_alt(img,img_lbl,alt_thresh):
    # Filter out all connected components in label image whose corona altitute is above backgrount threshold
    # /!\ input is a label image, but output is a binary image (where CCs of label image have been filtered)
    img_rp = skim.measure.regionprops(img_lbl)
    wh = 1
    for rp in img_rp: # for all connected components in label image
        bb = rp.bbox # boundingbox of connected component
        xmin = max(0,bb[0]-wh)
        xmax = min(bb[2]+wh+1,img_lbl.shape[0])
        ymin = max(0,bb[1]-wh)
        ymax = min(bb[3]+wh+1,img_lbl.shape[1])
        img_lbl_crop = img_lbl[xmin:xmax,ymin:ymax] # select only crop in label image corresponding to CC
        reg = img_lbl_crop==rp.label # Get binary region corresponding to CC
        corona_reg = get_corona_region(reg) # Get corona of this region (crop is for fast processing)
        img_crop = img[xmin:xmax,ymin:ymax] # select only crop in grayscale image corresponding to CC
        if alt_thresh >= 20:
            if img_crop[corona_reg].max() > 1.1*alt_thresh:
                # Remove this CC if max of its corona is slightly above background threshold
                img_lbl[img_lbl==rp.label] = 0
        else:
            if img_crop[corona_reg].max() > 2*alt_thresh:
                # Remove this CC if max of its corona is slightly above background threshold
                img_lbl[img_lbl==rp.label] = 0
    return img_lbl > 0 

def detect_candidate_cosmic(images, new_star, imgs_complete, area_thresh=[3,25],pfa_bck=0.01,verbose=False, filter_corona=False):
    # Detect all potential cosmic rays in given image
    # All passed parameters have the following default values:
    #    - area_thresh = [3,25] -> detected region should have a size between 3 and 25 pixels
    #    - pfa_bck=0.01 -> pfa for bakground estimation set by default to 1%
    #    - verbose=False -> set to True if you want to display some infos during detection
    
    # compute max-tree, altitude, area and gradient attributes
    counter = 0
    corrupted_index = []
    detect_clean = defaultdict(list)
    alt_thresh = defaultdict(list)
    #for i in range(len(images)):
    for i, j in enumerate(new_star):
        print('image number: ', i, j)
        graph = hg.get_4_adjacency_graph(images[j].shape)
        mt,alt = hg.component_tree_max_tree(graph,images[j]) # max-tree + altitude (gray level) attribue
        attr_area = hg.attribute_area(mt) # area attribute
        sel_grad = skim.morphology.square(3) #disk(3)
        img_grad_morpho = skim.morphology.dilation(images[j],sel_grad) - images[j]
        weight_grad = hg.weight_graph(graph,img_grad_morpho,hg.WeightFunction.mean) #average gradient
        attr_strength = hg.attribute_contour_strength(mt,weight_grad) # gradient strength attribute
        #print('attr_strenght: ', attr_strength)
        if images[j].dtype=='uint8':
            # if image is in uint8 format, gradient strength of detected region must be >= 3 (arbitrary)
            strength_min = 2
        else:
            # if image is other format, gradient strength of detected region must be >= 6 (also arbitrary)
            strength_min = 6

        # estimate background stats / background area
        try:
            mu_bck,std_bck,alt_thresh[j] = estimate_bck_stats(images[j], imgs_complete[j[:-1]+'IMG'], pfa=pfa_bck, debug=True) # stats
            
        except Exception:
            corrupted_index.append(j)
            counter +=1
          #  print('corrupted image number: ', i)
            continue
        #alt_thresh = 10
        bck = images[j] <= alt_thresh[j] # background binary image
        if verbose:
            #print stats infos of estimated background
            
            print('Estimated background mean: %1.3f'%mu_bck)
            print('Estimated background std: %1.3f'%std_bck)
            print('Estimated background threshold for pfa=%1.2f: %1.3f'%(pfa_bck,alt_thresh[j]))

        # filter max-tree with altitude, area and gradient constraints
        cond_alt = alt <= alt_thresh[j] # candidate region must have max gray scale value > background threshold
        cond_area = (attr_area <= area_thresh[0])|(attr_area>area_thresh[1]) # also satisfy the size constraints
        cond_strength = attr_strength<=strength_min # and the gradient strength constraints
        # Remove all nodes from max-tree that do not satisfy the previous constraints
        img_detect = hg.reconstruct_leaf_data(mt,alt,cond_alt|cond_area|cond_strength)

        # turn reconstructed image into binary image
        img_detect_bw = img_detect > alt[-1]
        img_lbl = skim.measure.label(img_detect_bw)
        # Remove CCs at less that 5 pixels of the image borders
        detect_clean[j] = revome_border_CCs(img_lbl)

        # Filter CCs whose outer border (corona) is higher than background threshold
        img_lbl = skim.measure.label(detect_clean[j])
        if filter_corona:
            detect_clean[j] = filter_corona_alt(images[j],img_lbl,alt_thresh[j])
        if verbose:
            img_lbl = skim.measure.label(detect_clean[j])
            print('%d detected candidate cosmics'%img_lbl.max())

    return detect_clean,alt_thresh, corrupted_index, counter # return binary detection image + background threshold

def regionprops(detect_clean_im):
    
    #area_thresh_min = 3
    images_true = defaultdict(list)
    for i in detect_clean_im:
        #print('ok',i)
        images_true[i].append(detect_clean_im[i])

    rp_image_final = defaultdict(list)
    for i in images_true:
        #print('yes', i)
        lbl = measure.label(images_true[i][0])
        rp_image_final[i].append(measure.regionprops(lbl))

    return rp_image_final
             
def conservative_upscaling(arr, scale):
    """
    Upscale an array using integer scales while keeping original
    values of the input array. The last values of every dimension
    will be the same as the neighboring ones.
 
    Note that not all values will be conserved for an axis with an
    odd dimension that is upscaled to an output dimension that is
    also odd.
 
    Parameters
    ----------
    arr : array_like
        The array to upscale.
    scale : int or array_like of int
        The scale or scales to apply to the input array.
    """
 
    arr = np.array(arr, dtype='float')
    input_shape = arr.shape
    print(input_shape)
    n = len(input_shape)
 
    if type(scale) == int:
        scale = (scale,) * n
   
    try:
        for s in scale:
            if type(s) != int:
                raise ValueError('The scale variable must an integer or an array-like of integers.')
    except:
        raise ValueError('The scale variable must an integer or an array-like of integers.')
   
    if n != len(scale):
        raise ValueError(f'Cannot broadcast array shape of dimension {n} and given scales of length {len(scale)}.')
   
    for s in scale:
        if s < 1:
            raise ValueError(f'Scales must be of 1 or higher, got {scale} instead.')
   
    pad = []
    crop = []
 
    output_shape = tuple([i*s for i, s in zip(input_shape, scale)])
    print(output_shape)
 
    for i, o in zip(input_shape, output_shape):
 
        in_is_even = i % 2 == 0
        out_is_even = o % 2 == 0
       
        pad += [in_is_even]
        crop += [out_is_even]
   
    do_crop = np.any(crop)
    do_pad = np.any(pad) or do_crop
 
    if do_pad:
        padding = tuple([(0, 1) if p else (0, 2 * int(c)) for p, c in zip(pad, crop)])
        arr = np.pad(arr, padding, 'edge')
        input_shape = arr.shape
 
    if do_crop:
        output_shape = tuple([s + 1 + 2 * (1 - int(p)) if c else s for s, c, p in zip(output_shape, crop, pad)])
 
    zoom_ratio = tuple([o / i for o, i in zip(output_shape, input_shape)])
    arr = zoom(arr, zoom_ratio, order=1, mode='mirror', grid_mode=False)
 
    if do_crop:
        slices = []
        for s, c, p in zip(output_shape, crop, pad):
            if c:
                slices += [slice(0, s - 1 - 2 * (1 - int(p)))]
            else:
                slices += [slice(0, s)]
        arr = arr[tuple(slices)]
   
    return arr

def build_dataframe(xyz, rp_final, latitude, distance_qmpf, time, images8_dict, new_stars, mean_energy, distance_enceladus):
    
    x = defaultdict()
    y = defaultdict()
    z = defaultdict()
    for i in xyz:
        x[i]= xyz[i][0][0][0][0]
        y[i] = xyz[i][0][0][0][1]
        z[i] = xyz[i][0][0][0][2]
        
    import datetime as dt
    date_format = '%Y-%jT%H:%M:%S'
    distance_time_cosmics = defaultdict(list)
    for j, i in enumerate(new_stars):
        if rp_final[i] != []:
            #print(i)
            distance_time_cosmics[i]= [dt.datetime.strptime(time[i[:-1]+'QMPF'][0][:-4], date_format), distance_qmpf[i[:-1]+'QMPF'], len(rp_final[i][0]), latitude[i[:-1]+'QMPF'], x[i[:-1]+'QMPF'], y[i[:-1]+'QMPF'], z[i[:-1]+'QMPF'], mean_energy[i[:-1]+'QMPF'], distance_enceladus[i[:-1]+'QMPF']]
    
    df_histo=pd.DataFrame.from_dict(distance_time_cosmics, orient='index', columns = ['time','distance','cosmics','latitude', 'x','y','z','energy', 'dist_enc'])
    
    df_img = defaultdict(list)
    for i, j in enumerate(images8_dict):
        if (j[:-4]+'1' in df_histo.index) and (j[:-4]+'2' in df_histo.index) and (j[:-4]+'3' in df_histo.index) and (j[:-4]+'4' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-4]+'1'], df_histo['distance'][j[:-4]+'1'], df_histo['cosmics'][j[:-4]+'1']+df_histo['cosmics'][j[:-4]+'2']+df_histo['cosmics'][j[:-4]+'3']+df_histo['cosmics'][j[:-4]+'4'],df_histo['latitude'][j[:-4]+'1'],df_histo['x'][j[:-4]+'1'],df_histo['y'][j[:-4]+'1'],df_histo['z'][j[:-4]+'1'],df_histo['energy'][j[:-4]+'1'], df_histo['dist_enc'][j[:-4]+'1']]
        elif (j[:-4]+'2' in df_histo.index) and (j[:-4]+'3' in df_histo.index) and (j[:-4]+'4' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-4]+'2'], df_histo['distance'][j[:-4]+'2'], df_histo['cosmics'][j[:-4]+'2']+df_histo['cosmics'][j[:-4]+'3']+df_histo['cosmics'][j[:-4]+'4'],df_histo['latitude'][j[:-4]+'2'], df_histo['x'][j[:-4]+'2'],df_histo['y'][j[:-4]+'2'],df_histo['z'][j[:-4]+'2'],df_histo['energy'][j[:-4]+'2'], df_histo['dist_enc'][j[:-4]+'2']]
        elif (j[:-4]+'1' in df_histo.index) and (j[:-4]+'3' in df_histo.index) and (j[:-4]+'4' in df_histo.index):   #print(j,i)
            df_img[j]= [df_histo['time'][j[:-4]+'1'], df_histo['distance'][j[:-4]+'1'], df_histo['cosmics'][j[:-4]+'1']+df_histo['cosmics'][j[:-4]+'3']+df_histo['cosmics'][j[:-4]+'4'],df_histo['latitude'][j[:-4]+'3'], df_histo['x'][j[:-4]+'3'],df_histo['y'][j[:-4]+'3'],df_histo['z'][j[:-4]+'3'], df_histo['energy'][j[:-4]+'3'],df_histo['dist_enc'][j[:-4]+'3']]
        elif (j[:-4]+'2' in df_histo.index) and (j[:-4]+'1' in df_histo.index) and (j[:-4]+'4' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-4]+'1'], df_histo['distance'][j[:-4]+'1'], df_histo['cosmics'][j[:-4]+'1']+df_histo['cosmics'][j[:-4]+'2']+df_histo['cosmics'][j[:-4]+'4'],df_histo['latitude'][j[:-4]+'4'], df_histo['x'][j[:-4]+'4'],df_histo['y'][j[:-4]+'4'],df_histo['z'][j[:-4]+'4'], df_histo['energy'][j[:-4]+'4'], df_histo['dist_enc'][j[:-4]+'4']]
        elif (j[:-4]+'2' in df_histo.index) and (j[:-4]+'3' in df_histo.index) and (j[:-4]+'1' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-4]+'1'], df_histo['distance'][j[:-4]+'1'], df_histo['cosmics'][j[:-4]+'1']+df_histo['cosmics'][j[:-4]+'2']+df_histo['cosmics'][j[:-4]+'3'],df_histo['latitude'][j[:-4]+'1'], df_histo['x'][j[:-4]+'1'],df_histo['y'][j[:-4]+'1'],df_histo['z'][j[:-4]+'1'], df_histo['energy'][j[:-4]+'1'], df_histo['dist_enc'][j[:-4]+'1']]
        elif (j[:-4]+'1' in df_histo.index) and (j[:-4]+'2' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-4]+'1'], df_histo['distance'][j[:-4]+'1'], df_histo['cosmics'][j[:-4]+'1']+df_histo['cosmics'][j[:-4]+'2'],df_histo['latitude'][j[:-4]+'1'],df_histo['x'][j[:-4]+'1'],df_histo['y'][j[:-4]+'1'],df_histo['z'][j[:-4]+'1'], df_histo['energy'][j[:-4]+'1'], df_histo['dist_enc'][j[:-4]+'1']]
        elif (j[:-4]+'1' in df_histo.index) and (j[:-4]+'3' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-4]+'1'], df_histo['distance'][j[:-4]+'1'], df_histo['cosmics'][j[:-4]+'1']+df_histo['cosmics'][j[:-4]+'3'],df_histo['latitude'][j[:-4]+'1'], df_histo['x'][j[:-4]+'1'],df_histo['y'][j[:-4]+'1'],df_histo['z'][j[:-4]+'1'], df_histo['energy'][j[:-4]+'1'], df_histo['dist_enc'][j[:-4]+'1']]
        elif (j[:-4]+'1' in df_histo.index) and (j[:-4]+'4' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-4]+'1'], df_histo['distance'][j[:-4]+'1'], df_histo['cosmics'][j[:-4]+'1']+df_histo['cosmics'][j[:-4]+'4'],df_histo['latitude'][j[:-4]+'1'], df_histo['x'][j[:-4]+'1'],df_histo['y'][j[:-4]+'1'],df_histo['z'][j[:-4]+'1'], df_histo['energy'][j[:-4]+'1'], df_histo['dist_enc'][j[:-4]+'1']]
        elif (j[:-4]+'2' in df_histo.index) and (j[:-4]+'3' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-4]+'2'], df_histo['distance'][j[:-4]+'2'], df_histo['cosmics'][j[:-4]+'3']+df_histo['cosmics'][j[:-4]+'2'],df_histo['latitude'][j[:-4]+'2'], df_histo['x'][j[:-4]+'2'],df_histo['y'][j[:-4]+'2'],df_histo['z'][j[:-4]+'2'], df_histo['energy'][j[:-4]+'2'], df_histo['dist_enc'][j[:-4]+'2']]
        elif (j[:-4]+'2' in df_histo.index) and (j[:-4]+'4' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-4]+'2'], df_histo['distance'][j[:-4]+'2'], df_histo['cosmics'][j[:-4]+'2']+df_histo['cosmics'][j[:-4]+'4'],df_histo['latitude'][j[:-4]+'2'], df_histo['x'][j[:-4]+'2'],df_histo['y'][j[:-4]+'2'],df_histo['z'][j[:-4]+'2'], df_histo['energy'][j[:-4]+'2'], df_histo['dist_enc'][j[:-4]+'2']]
        elif (j[:-4]+'3' in df_histo.index) and (j[:-4]+'4' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-4]+'3'], df_histo['distance'][j[:-4]+'3'], df_histo['cosmics'][j[:-4]+'3']+df_histo['cosmics'][j[:-4]+'4'],df_histo['latitude'][j[:-4]+'3'], df_histo['x'][j[:-4]+'3'],df_histo['y'][j[:-4]+'3'],df_histo['z'][j[:-4]+'3'], df_histo['energy'][j[:-4]+'3'], df_histo['dist_enc'][j[:-4]+'3']]
        elif (j[:-4]+'1' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-4]+'1'], df_histo['distance'][j[:-4]+'1'], df_histo['cosmics'][j[:-4]+'1'],df_histo['latitude'][j[:-4]+'1'], df_histo['x'][j[:-4]+'1'],df_histo['y'][j[:-4]+'1'],df_histo['z'][j[:-4]+'1'], df_histo['energy'][j[:-4]+'1'], df_histo['dist_enc'][j[:-4]+'1']]
        elif (j[:-4]+'2' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-4]+'2'], df_histo['distance'][j[:-4]+'2'], df_histo['cosmics'][j[:-4]+'2'],df_histo['latitude'][j[:-4]+'2'], df_histo['x'][j[:-4]+'2'],df_histo['y'][j[:-4]+'2'],df_histo['z'][j[:-4]+'2'], df_histo['energy'][j[:-4]+'2'], df_histo['dist_enc'][j[:-4]+'2']]
        elif (j[:-4]+'3' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-4]+'3'], df_histo['distance'][j[:-4]+'3'], df_histo['cosmics'][j[:-4]+'3'],df_histo['latitude'][j[:-4]+'3'], df_histo['x'][j[:-4]+'3'],df_histo['y'][j[:-4]+'3'],df_histo['z'][j[:-4]+'3'], df_histo['energy'][j[:-4]+'3'], df_histo['dist_enc'][j[:-4]+'3']]
        elif (j[:-4]+'4' in df_histo.index):
            df_img[j]= [df_histo['time'][j[:-4]+'4'], df_histo['distance'][j[:-4]+'4'], df_histo['cosmics'][j[:-4]+'4'],df_histo['latitude'][j[:-4]+'4'], df_histo['x'][j[:-4]+'4'],df_histo['y'][j[:-4]+'4'],df_histo['z'][j[:-4]+'4'],df_histo['energy'][j[:-4]+'4'], df_histo['dist_enc'][j[:-4]+'4']]
            #print(j)
    df_final = pd.DataFrame.from_dict(df_img, orient='index', columns = ['time','distance','cosmics','latitude','x','y','z', 'energy', 'dist_enc'])
    
    df_time = df_final.drop(labels = ['distance','latitude'], axis=1)
    df_dist = df_final.drop(labels = ['time','latitude'], axis=1)
    df_lat = df_final.drop(labels = ['time','distance'], axis=1)
    
    return df_final, df_dist, df_time, df_lat

def energy_estimation(images8_list,rp_final, detected_images, normal_QE = True):
    import os
    gain_dic = defaultdict()
    for file in os.listdir('/lre/home/gquaglia/MyISSdir'):
        for line in open('/lre/home/gquaglia/MyISSdir/' + file, "br"):
            if b'GAIN_MODE_ID' in line:
                a = str(line)
                # remove trailing newline
                gain_dic[file] = a[str(line).find('ELECTRONS PER DN')-3:str(line).find('ELECTRONS PER DN')-1]
    if normal_QE == True:
        Energy=defaultdict(list)
        for i in images8_list:
            for j in range(len(rp_final[i][0])):
                wh=2
                bb = rp_final[i][0][j].bbox # boundingbox of connected component
                xmin = max(0,bb[0]-wh)
                xmax = min(bb[2]+wh,detected_images[i].shape[0])
                ymin = max(0,bb[1]-wh)
                ymax = min(bb[3]+wh,detected_images[i].shape[1])
                img_lbl_crop = detected_images[i][xmin:xmax,ymin:ymax]
                back_mean = images8_list[i][xmin:xmax,ymin:ymax][detected_images[i][xmin:xmax,ymin:ymax]==0].mean()
                signal_intensity = images8_list[i][xmin:xmax,ymin:ymax][detected_images[i][xmin:xmax,ymin:ymax]>0].sum()
                #Energy[i].append((signal_intensity*int(gain_dic[i[:-1]+'IMG']))*(786.5/sp.constants.N_A)) #Energy in kJ
                Energy[i].append(((signal_intensity*int(gain_dic[i[:-1]+'IMG']))*100/35)*(786.5/sp.constants.N_A)*(6.2415E+15)) #Energy in MeV
    #786.5 = energy first ionization Si /// 6.2415E+15 = conversion kJ to MeV /// 100/35 Quantum efficiency ccd
    else:
        Energy=defaultdict(list)
        for i in images8_list:
            for j in range(len(rp_final[i][0])):
                wh=2
                bb = rp_final[i][0][j].bbox # boundingbox of connected component
                xmin = max(0,bb[0]-wh)
                xmax = min(bb[2]+wh,detected_images[i].shape[0])
                ymin = max(0,bb[1]-wh)
                ymax = min(bb[3]+wh,detected_images[i].shape[1])
                img_lbl_crop = detected_images[i][xmin:xmax,ymin:ymax]
                back_mean = images8_list[i][xmin:xmax,ymin:ymax][detected_images[i][xmin:xmax,ymin:ymax]==0].mean()
                signal_intensity = images8_list[i][xmin:xmax,ymin:ymax][detected_images[i][xmin:xmax,ymin:ymax]>0].sum()-(back_mean*len(detected_images[i][xmin:xmax,ymin:ymax][detected_images[i][xmin:xmax,ymin:ymax]>0]))
                #Energy[i].append((signal_intensity*int(gain_dic[i[:-1]+'IMG']))*(786.5/sp.constants.N_A)) #Energy in kJ
                Energy[i].append(((signal_intensity*int(gain_dic[i[:-1]+'IMG']))*100/35)*(786.5/sp.constants.N_A)*(6.2415E+15)) #Energy in MeV
    
    en_values = []
    for i in Energy:
        for j in Energy[i]:
            en_values.append(j)
            
    return gain_dic, Energy, en_values