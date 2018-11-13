#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import numpy as np



class Combinator:
    def __init__(self, dataset_path, img_cnn, img_layer, txt_model, txt_layer, output_path):
        self.dataset_path = dataset_path
        self.img_cnn = img_cnn
        self.img_layer = img_layer
        self.txt_model = txt_model
        self.txt_layer = txt_layer
        self.outpath = output_path

        self.txtrecord2img = {}
        self.txt_features = {}
        self.img_features = {}
        self.img_labels = {}

    def combine_dataset(self, traintest):

        print("Loading labels and paths of images:")
        immagini = []
        imgfile=self.dataset_path+"/"+traintest+".txt"
        with open(imgfile, 'rt') as f:
            for line in [line.strip() for line in f.readlines()]:
                cols = [col.strip() for col in line.split(' ', 3)]
                img, sent = cols
                immagini.append(os.path.basename(img))
                self.img_labels[os.path.basename(img)] = sent
        print(" images[]={}".format(len(immagini)))
        print(" labels[]={}".format(len(self.img_labels)))
        raw_input("push a button...")

        print("Loading visual features:")
        with open(os.path.join('{}/{}/'.format(visfeatures_path,self.img_cnn), '{}_{}.features.csv'.format(traintest, self.img_layer)), 'rt') as f:
            ind=0
            for line in [line.strip() for line in f.readlines()]:
                features = line.split(',')
                img=immagini[ind]
                ind += 1
                self.img_features[img] = features
        print(" img_features[]={}".format(len(self.img_features)))
        raw_input("push a button...")

        # read text training file to see which images had text
        print("Loading paths of images with text:")
        imgs_with_text = {}
        line_no = 0
        with open(os.path.join(txtonehot_path, '{}_{}.csv'.format(self.txt_model, traintest)), 'rt') as f:
            for line in [line.strip() for line in f.readlines()]:
                #reading (PATH - ONEHOT_TEXT)
                img, rest = line.split(',', 1)
                imgs_with_text[line_no] = img
                #print(traintest, ':', line_no, ':', '>{}<'.format(img))
                line_no += 1
                #CHECK
                if img not in self.img_features:
                    print('ERROR:', img, 'not in training images!')


        print(' Read', len(imgs_with_text), 'images with text')
        self.txtrecord2img = imgs_with_text
        imgs_with_text = imgs_with_text.values()     # have a list of images with text handy
        raw_input("push a button...")

        # read the text features
        # ... and generate neutral text features for missing texts
        print("Loading textual features of images with text:")
        missing_features = ''
        with open(os.path.join(txtfeatures_path, 'features_{}_{}_{}.csv'.format(self.txt_model,traintest, self.txt_layer)), 'rt') as f:
            for line in [line.strip() for line in f.readlines()]:
                #reading (Number - Features)
                record, features = line.split(',', 1)
                if not missing_features:    #we create neutral features for images with no text
                    feature_len = len(features.split(','))
                    missing_features = ','.join(['0'] * feature_len)
                self.txt_features[self.txtrecord2img[int(record)]] = features
        print(" txt_features[]={}".format(len(self.txt_features)))
        raw_input("push a button...")

        # fill the features of imgs with no text
        miss=0
        print("Adding neutral text features to images that do not have text:   neutral features contain {} zeros!!".format(len(missing_features)))
        for img in self.img_features:
            if img not in imgs_with_text:
                self.txt_features[img] = missing_features
                #print('Missing >{}<'.format(img))
                miss+=1
        print(" miss_txt={}".format(miss))

        print("CHECK:   {}+{}={}={}".format(len(self.txt_features),miss,len(self.txt_features)+miss,len(self.img_labels)))
        raw_input("push a button...")

        # now go through all
        print("FINAL DATASET...")
        suffix = '{}_{}_{}_{}'.format(self.img_cnn, self.img_layer, self.txt_model, self.txt_layer)
        with open(os.path.join(self.outpath, '{}_{}_combined_dataset.csv'.format(traintest, suffix)), 'wt') as f:
            for img, img_features in self.img_features.items():
                txt_features = self.txt_features[img]
                label = self.img_labels[img]
                img_features2print = ','.join(img_features)
                f.write('{},{},{},{},{},{}\n'.format(img,label, len(img_features), len(txt_features.split(',')), img_features2print, txt_features))

        print("DONE!!")


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('Usage: {} img_cnn img_layer txt_model txt_layer output_path'.format(sys.argv[0]))
        sys.exit()

    img_cnn = sys.argv[1]
    img_layer = sys.argv[2]
    txt_model = sys.argv[3]
    txt_layer = sys.argv[4]
    output_path = sys.argv[5]

    ds_path = "../dataset"  #folder for dataset labels (train.txt and test.txt files)
    visfeatures_path="../results/visual/img_features"
    txtfeatures_path="../results/textual/txt_features"
    txtonehot_path="../results/textual/onehot"

    for ds in 'train', 'test':
        print('Generating', ds, img_layer, txt_model, txt_layer)
        c = Combinator(dataset_path=ds_path, img_cnn=img_cnn, img_layer=img_layer, txt_model=txt_model, txt_layer=txt_layer, output_path=output_path)
        c.combine_dataset(ds)


