# Deep learning based detection and classification of small bright sources on images from the Cassini mission

This is the code if you want to use a personalized detection algorithm working on images from the Cassini Imaging Science Subsystem (ISS) database. The network is able to detect and classify cosmic rays, stars and satellites on NAC, 8bit images with an exposure duration <= 1.2 seconds. 

All the codes have been tested on macOS 13.0.1 using Python 3.9. 

## Dataset download

The dataset used in our work can be downloaded at the following link: [PDS Planetary Data System](https://pds-imaging.jpl.nasa.gov/search/?fq=-ATLAS_THUMBNAIL_URL%3Abrwsnotavail.jpg&q=*%3A*), from the PDS Image Atlas. Follow the instructions on their site and download all the NAC images you need from the Cassini mission.
You have to convert the images in one of the following formats: ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng'] 

## Download Yolov5

Download yolov5 from here: [Yolov5](https://github.com/ultralytics/yolov5), following their instructions. 

## Data preparation

Once you have all the images downloaded you have to filter out all the images which are not 8bit and have an exposure duration greater than 1.2 seconds. 

The network is trained on 532x532 images and the original images are 1024x1024, so you have to split them in 4 with a 20 pixels margin.

Then you have to convert the images in one of the following formats: ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng'] 


You can perform all these operations using the     FILE PER OPERAZIONI          file, you have to change the path where the converted images are stored in your computer. You have to have the file   LIBRERIA      saved in the same folder where you have     FILE PER OPERAZIONI     . 


## Infering the network

Save the weights from the 

Now you can infer the network on the images with this command:

```
python detect.py --source /path/to/your/images --weights /path/to/the/weights/last.pt --batch 32 --save-txt

```

It will save a a .txt file for each image in the following format:

```
class_id center_x center_y width height
```

Class_id is 0 for cosmic rays, 1 for stars and 2 for satellites.

The center_x, center_y, width, height, are normalized on the image dimension, therefore to have the actual values on the image you have to multiply them for 532. 






