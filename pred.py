import cv2
import numpy as np
import h5py
import time
import os
import math

import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import load_model, Input, Model
from glob import glob

model1 = load_model('upmodel1.h5')
model2 = load_model('upmodel2.h5')

model1.summary()
model2.summary()

dos = glob('exemple\\*') #please enter the path of the file containing image
i = 0
for img_path in tqdm(dos):
    try:
        
        imgo = cv2.imread(img_path)
        i+=1

        w,h = imgo.shape[1], imgo.shape[0]

        total = w*h
        maximum = 1500*1500
        
        if total>maximum: #if image is too big, divideimage in multiple little image to pred and reconstruct it later
            ratio = maximum/total

            nb = 2

            while(ratio<0.50):
                ratio*=2
                nb*= 2

            w = w//nb
            h = h//nb

             
            comp_img = []

            for xs in range(1,nb+1):
                for ys in range(1,nb+1):
                    img = imgo[int(h*ys-h):int(h*ys), int(w*xs-w):int(w*xs)]                

                    pred = np.expand_dims(img,axis=0)

                    x = model1.predict(pred/255)
                    new_img = model2.predict(x)

                    comp_img.append(new_img[0])
            
            comp_img = np.array(comp_img)
            new_img = np.zeros((new_img.shape[1]*nb,new_img.shape[2]*nb,3))

            h *= new_img.shape[0]//imgo.shape[0]
            w *= new_img.shape[1]//imgo.shape[1]
            n = 0

            #reconstruct image
            for xs in range(1,nb+1):
                for ys in range(1,nb+1):
                    new_img[h*ys-h :h*ys, w*xs-w :w*xs] = comp_img[n]
                    n+=1                
                    
            cv2.imwrite('out\\'+str(i)+'.jpg',new_img*255) #please enter your output file


        else: #if image is little enought just pred the whole image

            pred = np.expand_dims(imgo,axis=0)

            x = model1.predict(pred/255)
            new_imgo = model2.predict(x)
        
            cv2.imwrite('out\\'+str(i)+'.jpg',new_imgo[0]*255) #please enter your output file

    except:
        print('error')

        
        





