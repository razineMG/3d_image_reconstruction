import cv2
import numpy as np
import pandas as pd

path_light_position='/mnt/c/Users/gr009/desktop/s3/vision/3d_image_reconstruction/objet1PNG_SII_VISION/light_directions.txt'
path_light_intens = '/mnt/c/Users/gr009/desktop/s3/vision/3d_image_reconstruction/objet1PNG_SII_VISION/light_intensities.txt'
mask_path="/mnt/c/Users/gr009/desktop/s3/vision/3d_image_reconstruction/objet1PNG_SII_VISION/mask.png"
images_path='/mnt/c/Users/gr009/desktop/s3/vision/3d_image_reconstruction/objet1PNG_SII_VISION/'

def load_lightSources(path):
    data = pd.read_csv(path, sep = " ", header=None)
    lightPositionMatrix = data.to_numpy()
    return lightPositionMatrix

#print(load_lightSources(path_light))

def load_intensSourses(path):
    data = pd.read_csv(path, sep = " ", header=None)
    lightIntens = data.to_numpy()
    return lightIntens

#print(type(load_intensSourses(path_light_intens)[0][0]))


def load_objMask(path):
    mask=cv2.imread(path,0)
    h,w=mask.shape
    for y in range(h):
        for x in range(w):
            if mask[y][x] == 255:
                mask[y][x]=1
    return mask

#print(load_objMask(mask_path)[200])

def load_images(path):
    image_table=[]
    for i in range(3):
        real_path=path+format(i+1,'03d')+'.png'
        print(real_path)
        image=cv2.imread(real_path,-1)
        h,w,c=image.shape
        norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #print(type(norm_image[0][0][0]))
        print(norm_image[0][200])
        light_intens_matrix=load_intensSourses(path_light_intens)
        for y in range(h): # divide pixel each pixel by intensity source
            for x in range(w):
                norm_image[y][x][0]/=light_intens_matrix[0][0]
                norm_image[y][x][1]/=light_intens_matrix[0][1]
                norm_image[y][x][2]/=light_intens_matrix[0][2]
        for y in range(h): # convert image to grey level\
            for x in range(w):
                norm_image[y][x][0]*=0.3
                norm_image[y][x][1]*=0.59
                norm_image[y][x][2]*=0.11   
        image_table.append(norm_image.reshape(-1))  
        
    return image_table
    

print(load_images(images_path))



