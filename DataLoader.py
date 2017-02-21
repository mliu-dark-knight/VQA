import glob
import math
import numpy as np
import random
import time

from scipy import misc
from keras.utils.np_utils import to_categorical

class ImageLoadingGentor:
    def __init__(self, images_path, labels_path, output_x, output_y, num_classes, train_batch_size=32, val_batch_size=32, val_prop=0.2):
        assert 0 <= val_prop <= 1
        self.images_path = images_path
        self.labels_path = labels_path
        self.output_x = output_x
        self.output_y = output_y
        self.num_classes = num_classes

        self.val_prop = val_prop
        self.val_batch_size = val_batch_size
        self.rnded_srcimg_fname_list_val = []
        self.rnded_lblimg_fname_list_val = []
        self.imagearr_val = np.empty((self.val_batch_size, self.output_x, self.output_y, 3))
        self.labels_val = np.empty((self.val_batch_size, self.output_x, self.output_y, 3))
        self.val_currpos = 0

        self.train_batch_size = train_batch_size
        self.rnded_srcimg_fname_list_train = []
        self.rnded_lblimg_fname_list_train = []
        self.imagearr_train = np.empty((self.train_batch_size, self.output_x, self.output_y, 3))
        self.labels_train = np.empty((self.train_batch_size, self.output_x, self.output_y, 3))
        self.train_currpos = 0

        self.next_epoch()

    def next_train_batch(self):
        for i in range(0, self.train_batch_size):
            if self.train_currpos >= len(self.rnded_srcimg_fname_list_train):
                print("Repeating Train Split")
                self.train_currpos = 0

            curr_srcimg_fname = self.rnded_srcimg_fname_list_train[self.train_currpos]
            #curr_lblimg_fname = self.rnded_lblimg_fname_list_train[self.train_currpos]
            image = misc.imresize(misc.imread(curr_srcimg_fname, mode='RGB'), (self.output_x, self.output_y, 3), 'bicubic')
            image = np.array(image, dtype=np.float32, ndmin=4)

            image[:, :, 0] -= 103.939
            image[:, :, 1] -= 116.779
            image[:, :, 2] -= 123.68
            self.imagearr_train[i] = image
            #self.labels_train[i] = to_categorical(nb_classes=self.num_classes)

            self.train_currpos += 1

        return self.imagearr_train

    def next_val_batch(self):
        for i in range(0, self.val_batch_size):
            if self.val_currpos >= len(self.rnded_srcimg_fname_list_val):
                print("Repeating Val Split")
                self.val_currpos = 0

            curr_srcimg_fname = self.rnded_srcimg_fname_list_val[self.val_currpos]
            #curr_lblimg_fname = self.rnded_lblimg_fname_list_val[self.val_currpos]
            image = misc.imresize(misc.imread(curr_srcimg_fname, mode='RGB'), (self.output_x, self.output_y, 3), 'bicubic')
            image = np.array(image, dtype=np.float32, ndmin=4) / 255

            image[:, :, 0] -= 103.939
            image[:, :, 1] -= 116.779
            image[:, :, 2] -= 123.68
            self.imagearr_val[i] = image
            #self.labels_val[i] = to_categorical(nb_classes=self.num_classes)

            self.val_currpos += 1

        return self.imagearr_val

    def next_epoch(self):
        print("Regenerating Train/Val Split")
        rnded_srcimg_fname_list = glob.glob(self.images_path)
        rnded_lblimg_fname_list = glob.glob(self.labels_path)
        random.shuffle(rnded_srcimg_fname_list)
        random.shuffle(rnded_lblimg_fname_list)

        self.rnded_srcimg_fname_list_val = \
            rnded_srcimg_fname_list[:math.floor(self.val_prop * len(rnded_srcimg_fname_list))]
        self.rnded_lblimg_fname_list_val = \
            rnded_lblimg_fname_list[:math.floor(self.val_prop * len(rnded_lblimg_fname_list))]
        self.rnded_srcimg_fname_list_train = \
            rnded_srcimg_fname_list[math.floor(self.val_prop * len(rnded_srcimg_fname_list)):]
        self.rnded_lblimg_fname_list_train = \
            rnded_lblimg_fname_list[math.floor(self.val_prop * len(rnded_lblimg_fname_list)):]

if __name__ == "__main__":
    starttime = time.time()
    GTAloader = ImageLoadingGentor("/media/victor/Data/gtadata/images/*.png", "/media/victor/Data/gtadata/labels/*.png", 224, 224, 1000, val_prop=0)
    print(str(time.time()-starttime))
    for i in range(0, 100):
        starttime = time.time()
        GTAloader.next_train_batch()
        #print(GTAloader.next_train_batch())
        print(str(time.time() - starttime))
