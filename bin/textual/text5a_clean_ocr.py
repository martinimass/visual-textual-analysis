#!/usr/bin/env python
from __future__ import print_function
import sys
import os
from collections import defaultdict

srcfolder = "../results/textual/"   #path for train4 and test4 folders
labelfolder="../dataset"            #path for train.txt and test.txt files (textual labels)


class Cleaner:
    def __init__(self, folder, model):
        self.folder = folder
        self.model = model
        self.model_suffix = '.all.{}.txt'.format(self.model)
        self.num_records = 0
        self.all_imgs = defaultdict(dict)
        self.all_texts = {}
        self.ignore_texts = set()

    def read_labels(self, traintest):
        label_csv = traintest+'.txt'

        self.wd = os.path.join(self.folder, traintest+"4")

        with open(os.path.join(labelfolder, label_csv), 'rt') as f:
            cont_nolabel=0
            for line in [line.strip() for line in f.readlines() if line.strip()]:
                cols = [col.strip() for col in line.split(' ')]
                img = cols[0]
                img=os.path.split(img)[1]
                textual_lbl = cols[1]
                txtf = os.path.join(self.wd, img + self.model_suffix)
                if not os.path.exists(txtf):
                    print('no', txtf)
                    cont_nolabel+=1
                    continue
                # read text
                text = ''
                with open(txtf, 'rt') as f:
                    text = f.readline().strip()[:140]
                self.all_imgs[traintest][img] = (textual_lbl, text)
                if text not in self.all_texts:
                    self.all_texts[text] = (textual_lbl, img)
                else:
                    existing_lbl, existing_img = self.all_texts[text]
                    if  existing_lbl != textual_lbl:
                        print ('Conflict', traintest, img, text, textual_lbl, self.all_texts[text])
                        self.ignore_texts.add(text)
            print("{} - nolabel={}".format(traintest,cont_nolabel))

    def write_clean_labels(self, traintest):
        label_csv = traintest+'_{}_clean.csv'.format(self.model)
        
        #self.wd = os.path.join(self.folder, traintest)
        with open(os.path.join(self.folder, label_csv), 'wt') as f:
            for img in self.all_imgs[traintest]:
                lbl, text = self.all_imgs[traintest][img]
                if text in self.ignore_texts:
                    print('Ignoring', traintest, text)
                    continue
                f.write('{},{}\n'.format(img, lbl))

if __name__ == '__main__':
    
    for ocr_model in ["char2","dict2"]:

		datasets = ['train', 'test']
		all_texts = {}     # record label for texts
		ignore_imgs = []

		# filter out only duplicates with conflicting labels
		cleaner = Cleaner(srcfolder, ocr_model)
		for ds in datasets:
		    cleaner.read_labels(ds)
		for ds in datasets:
		    cleaner.write_clean_labels(ds)

		print('Texts with conflicts:', len(cleaner.ignore_texts))
		print(','.join(cleaner.ignore_texts))



