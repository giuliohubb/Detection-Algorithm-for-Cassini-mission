# Dataset download

The dataset used in our work can be downloaded at the following link: [PDS Planetary Data System](https://pds-imaging.jpl.nasa.gov/search/?fq=-ATLAS_THUMBNAIL_URL%3Abrwsnotavail.jpg&q=*%3A*), from the PDS Image Atlas. Follow the instructions on their site and download all the NAC images you need from the Cassini mission.
You have to convert the images in one of the following formats: ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng'] 

# Download Yolov5

Download yolov5 from here: [Yolov5](https://github.com/ultralytics/yolov5), following their instructions. 

Once you have it downloaded you can infer the model on your images using the following code:

'''
 python detect.py --source /path/to/your/images --weights /path/where/you/saved/model/last.pt 

 '''
