import numpy
import skimage.io, skimage.color, skimage.feature
import os
import pickle

fruits = ["apple","avocado","papaya", "raspberry", "mango", "lemon"]
# shape 4 class=1,962
#shape 6 class = 2881
dataset_features = numpy.zeros(shape=(2881, 360))
outputs = numpy.zeros(shape=(2881))

idx = 0
class_label = 0
for fruit_dir in fruits:
    curr_dir = os.path.join(os.path.sep, fruit_dir)
    all_imgs = os.listdir(os.getcwd()+curr_dir)
    for img_file in all_imgs:
        if img_file.endswith(".jpg"): # Ensures reading only JPG files.
            #open image as pixel array
            fruit_data = skimage.io.imread(fname=os.path.sep.join([os.getcwd(), curr_dir, img_file]), as_gray=False)
            #convert color image from rgb to hsv (illumination independent)
            fruit_data_hsv = skimage.color.rgb2hsv(rgb=fruit_data)
            #take the first value of each hsv matrix and make it histogram 
            hist = numpy.histogram(a=fruit_data_hsv[:, :, 0], bins=360)
            #take the hist value as our dataset feature for training the model
            dataset_features[idx, :] = hist[0]
            
            outputs[idx] = class_label
            idx = idx + 1
    #class label = per folder
    class_label = class_label + 1

with open("dataset_features_6.pkl", "wb") as f:
    pickle.dump(dataset_features, f)

with open("outputs_6.pkl", "wb") as f:
    pickle.dump(outputs, f)

print('Feature Extracted!')
