from __future__ import print_function
import os
import numpy as np
from keras.applications.vgg16 import *
from keras.preprocessing import image

img_dir = 'VQA/Images/mscoco/val2014/'
feature_dir = 'VQA/Features/mscoco/val2014/'


if __name__ == '__main__':
	model = VGG16(weights='imagenet', include_top=False)
	if not os.path.exists(feature_dir):
		os.makedirs(feature_dir)
	for filename in os.listdir(img_dir):
		img = image.img_to_array(image.load_img(img_dir + filename, target_size=(224, 224)))
		img = preprocess_input(np.expand_dims(img, axis=0))
		feature = np.reshape(model.predict(img), (7 * 7, 512))
		np.save(feature_dir + filename[:-4], feature)
