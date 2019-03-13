from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import AveragePooling2D, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm
import cv2
import time

class model_object():
    def __init__(self):

        #create or load models
        self.model1, self.model2 = self.create_model()

        #self.model1 = load_model('debmodel1.h5')
        #self.model2 = load_model('debmodel2.h5')



    def create_model(self):
        
        ############ IMAGE DEBLURING ############ 
        # I create 2 models to see data in between

        model1 = Sequential()
        model2 = Sequential()

        #### model 1

        model1.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding="same", input_shape=(None,None,3)))
        model1.add(Activation("relu"))

        model1.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding="same"))
        model1.add(Activation("relu"))

        model1.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding="same"))
        model1.add(Activation("relu"))

        #### model 2

        model2.add(Conv2DTranspose(32, kernel_size=(3,3), strides=(1,1), padding = "same", input_shape = (None,None,32)))
        model2.add(Activation("relu"))  

        model1.add(Conv2DTranspose(32, kernel_size=(3,3), strides=(1,1), padding="same"))
        model1.add(Activation("relu"))

        model2.add(Conv2D(3, kernel_size=(1,1), strides=(1,1), padding = "same"))
        model2.add(Activation("sigmoid"))

        return model1, model2


    def train(self, epochs = 20, batch_size = 32, X_train = [], Y_train = []):
        
        #creating combined model according to the current shape of the dataset

        img = Input(shape=(self.shape))
        x = self.model1(img)
        label = self.model2(x)

        self.combined = Model(img,label)
        self.combined.compile(loss='mse', optimizer=Adam())
        
        if epochs == None: #to train without epochs limit 
            epoch = 0
            while(True):
                epoch+=1
                print(epoch)

                self.combined.fit(X_train, Y_train, batch_size=batch_size, epochs=1)
                self.model1.save('debmodel1.h5')
                self.model2.save('debmodel2.h5')

        else: #to train with epochs limit
            self.combined.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)
            self.model1.save('debmodel1.h5')
            self.model2.save('debmodel2.h5')



if __name__ == '__main__':

    #create model
    model = model_object()

    #load dataset
    dos = glob('dataset\\*.npy')
    print(len(dos))

    #train model with every dataset 
    for i in range(len(dos)//2):

        X_train = np.load('dataset\\'+str(i)+'_X_train.npy')
        print('loaded X_train', i)
        Y_train = np.load('dataset\\'+str(i)+'_Y_train.npy')
        print('loaded Y_train', i)
    
        #train model
        model.shape = X_train.shape[1:]
        model.train(epochs=10, batch_size=1, X_train=X_train, Y_train=Y_train)


    

    
