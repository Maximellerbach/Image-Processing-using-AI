# Image-Processing-using-AI
in this repo, I create models to process image (upscale, debluring...)

##I used this dataset : http://ai.stanford.edu/~jkrause/cars/car_dataset.html

##you can use pretrained AI: 

upmodel1/2 are for upscaling image
debmodel1/2 are for debluring image

I used 2 models to visualize the output of the first model so I load both and create a combined one to train.
in pred.py, I just load both of them and use output of model1 as input of model2.
