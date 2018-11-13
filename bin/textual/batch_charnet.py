#!/usr/bin/env python

from __future__ import print_function
from numpy import argmax, array, dot, float32, mean, std, zeros

import skimage.io
import numpy as np



class CharNet():
    def __init__(self, architecture_file=None, weight_file=None, optimizer=None):
        # Generate mapping for softmax layer to characters
        output_str = '0123456789abcdefghijklmnopqrstuvwxyz '
        self.output = [x for x in output_str]
        self.L = len(self.output)

        # Load model and saved weights
        from keras.models import model_from_json
        if architecture_file is None:
            self.model = model_from_json(open('../../txt/reading-text-in-the-wild/CHAR2/char2_architecture.json').read())
        else:
            self.model = model_from_json(open(architecture_file).read())

        if weight_file is None:
            self.model.load_weights('../../txt/reading-text-in-the-wild/CHAR2/char2_weights.h5')
        else:
            self.model.load_weights(weight_file)

        if optimizer is None:
            from keras.optimizers import SGD
            optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def _rgb2gray(self,rgb):
        return dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    def _preprocess(self,img):
        from skimage.transform import resize
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = self._rgb2gray(img)
        img = resize(img, (32,100), order=1, preserve_range=True)
        img = array(img,dtype=float32) # convert to single precision
        img = (img - mean(img)) / ( (std(img) + 0.0001) )
        return img

    def classify_image(self,img):    
        img = self._preprocess(img)
        xtest = zeros((1,1,32,100))
        xtest[0,0,:,:] = img
        z = self.model.predict(xtest,verbose=0)[0,:]
        output_text = []
        for i in range(23):
            ind = argmax(z[i*self.L:(i+1)*self.L])
            output_text.extend(self.output[ind])
        return ''.join(output_text).rstrip(' ')


if __name__ == '__main__':
    import matplotlib.image as mpimg
    import sys
    import os
    import datetime


    crop_path = sys.argv[1]
    if not os.path.exists(crop_path+"3"):
        os.makedirs(crop_path+"3")
    print("CharNet - TEXT RECOGNITION of "+crop_path)

    now = datetime.datetime.now()
    print ("Current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))

    crop_files = [filn for filn in os.listdir(crop_path+"2") if filn.endswith('.crops_in_order.txt')]
    print("CharNet Loading")
    cnn_model = CharNet()
    
    cont=1
    for img_crops_filn in crop_files:
        lines = []
        print('{} of {} - Processing {}'.format(cont,len(crop_files),img_crops_filn))
        cont+=1
        with open(os.path.join(crop_path+"2", img_crops_filn), 'rt') as f:
            lines = [line.strip() for line in f.readlines()]
            for line in lines:
                cols = line.split(',')
                crop_filn = cols[0]
                print('   ', crop_filn)
                crop_filn2 = os.path.join(crop_path+"3", crop_filn)
                crop_filn = os.path.join(crop_path+"1", crop_filn)
                if os.path.isfile(crop_filn2 + '.char2.txt'):
                    print("      skipped..............")
                    continue
                img = mpimg.imread(crop_filn)
                result = cnn_model.classify_image(img)
                with open(crop_filn2 + '.char2.txt', 'wt') as f:
                    f.write('{}\n'.format(result))
	
    now = datetime.datetime.now()
    print ("Current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))
