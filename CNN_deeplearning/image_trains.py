import os
import numpy as np
import cv2

from glob import glob
from PIL import Image

#covert to image size(128,128) 
def convert_image_size(image_data):
    name=0
    for image in image_data:
        imag_pil = Image.open(image).convert("L")
        imag_norm = np.array(imag_pil)
        file_name=cv2.resize(imag_norm,dsize=(128,128))
        #save select
        cv2.imwrite('C:/Users/pjy/covid_analysis/total_test/0/{}.jpg'.format(name),file_name)
        name+=1
    return print(len(image_data))
