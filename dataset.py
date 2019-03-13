import numpy as np
from glob import glob
from tqdm import tqdm
import cv2

# create dataset
Y_train = []
X_train = []

dos = glob('..\\car_img\\*')
i=0

for path in tqdm(dos):
	img = cv2.imread(path)

	if img.shape[0]*img.shape[1]>=30000:
		img = cv2.resize(img,(150,100))
		x = cv2.blur(img,(2,2))
		X_train.append(x/255)

	if len(X_train)>= 2000:
		X_train = np.array(X_train)
		Y_train = np.array(Y_train)
		i+=1
		np.save('dataset\\'+str(i)+'_blurX_train.npy', X_train)
		np.save('dataset\\'+str(i)+'_blurY_train.npy', Y_train)

		print('saved X_train', i)
		print('saved Y_train', i)
		X_train = []
		Y_train = []

X_train = np.array(X_train)
Y_train = np.array(Y_train)
i+=1
np.save('dataset\\'+str(i)+'_X_train.npy', X_train)
np.save('dataset\\'+str(i)+'_Y_train.npy', Y_train)

print('saved X_train', i)
print('saved Y_train', i)
X_train = []
Y_train = []
