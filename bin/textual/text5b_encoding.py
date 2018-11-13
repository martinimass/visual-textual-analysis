#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import numpy as np
import random
from collections import Counter

srcfolder = "../results/textual"  #path for train4, test4 folders and "clean" file
outpath=srcfolder+"/onehot"        #folder for one-hot vector output
if not os.path.exists(outpath):
    os.makedirs(outpath)




class TextEncoder:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?'\""
    maxlen = 140
    alphalen = len(alphabet)

    def __init__(self):
        pass

    def encode(self, txt):
        data = np.zeros((self.maxlen, len(self.alphabet)))
        # reverse string and cap at 140 characters
        index = 0
        for character in txt[::-1][:self.maxlen]:
            if character in self.alphabet:
                onehotpos = self.alphabet.index(character)
                data[(index, onehotpos)] = 1.0
                # or else the character will be all zeroes
            index += 1
        return data

    def caffe_hot(self, data):
        s = ''
        for i in range(self.maxlen):
            for j in data[(i,)]:
                if j > 0.5:
                    s += '1'
                else:
                    s += '0'
        return s


class CaffeDataset:
    def __init__(self, folder, model, traintest, outpath, permutate=False):
        self.folder = folder
        self.outpath = outpath
        self.wd = os.path.join(folder, traintest+"4")
        self.model = model
        self.traintest = traintest
        self.permutate = permutate
        self.model_suffix = '.all.{}.txt'.format(self.model)
        self.num_records = 0
        self.input_files_lbls = []
        self.labels = None
        self.te = TextEncoder()
        self.csv_data = []
        self.category_counts = Counter()

    def read_labels(self):
        #Loading of .all..txt files and relative labels
        #carico in memoria i path dei file .all..txt delle immagini che hanno del testo, insieme alla propria label
        label_csv = self.traintest+'_{}_clean.csv'.format(self.model)

        with open(os.path.join(self.folder, label_csv), 'rt') as f:
            for line in [line.strip() for line in f.readlines() if line.strip()]:
                cols = [col.strip() for col in line.split(',')]
                img = cols[0]
                textual_lbl = cols[1]
                img = os.path.join(self.wd, img + self.model_suffix)
                if not os.path.exists(img):
                    #Redundant Control... checked in the cleaning phase
                    print('Warning: No text for', img)
                    continue
                self.input_files_lbls.append((img, int(textual_lbl)))
        self.labels = np.zeros(len(self.input_files_lbls), dtype='f4')
        random.shuffle(self.input_files_lbls)
        print("input_files_lbls={}".format(len(self.input_files_lbls)))
        raw_input()

    def read_data(self):
        #will be created:
        # - self.labels: textual label for every image
        # - self.csv_data: a triple fpr every image (  image path, onehot encoding,  textual label) 
        # - self.category_counts:   sum of encodings for every type of label
        cont=1
        for index, (filn, lbl) in enumerate(self.input_files_lbls):
            print("{} - Reading {}".format(cont,filn))
            cont+=1
            with open(filn, 'rt') as f:
                line = f.readline().strip()[:140]
                lines = [line]
                #skipped if permutate=False
                if self.permutate and self.traintest == 'train':
                    words = line.split(' ')
                    n_words = len(words)
                    perms = []
                    for i in range(1000):
                        line = ' '.join(words)
                        if line not in perms:
                            perms.append(line)
                        if len(perms) >= 100:
                            break
                        random.shuffle(words)
                    random.shuffle(lines)
                    lines = perms[:100]
                current_data = [self.te.encode(line) for line in lines]
                self.labels[index] = int(lbl)
                self.csv_data.append((filn.replace(self.model_suffix, ''), [self.te.caffe_hot(d) for d in current_data], int(lbl)))
                self.category_counts[int(lbl)] += len(current_data)
        raw_input()

    def write_caffe_data(self):
        csv_filn = os.path.join(self.outpath, '{}_{}.csv'.format(self.model, self.traintest))
        print('Generating', csv_filn)
        label_counter = Counter()
        random.shuffle(self.csv_data)

        #Dataset balancing
        if True or self.traintest == 'train':
            cap = min(self.category_counts.values())
        else:
            cap = 500000
            
        lines = []
        print("category_counts={}".format(self.category_counts.values()))
        print("cap={}".format(cap))
        
        with open(csv_filn, 'wt') as f:
            for img, data, label in self.csv_data:
                for data_str in data:
                    label_counter[label] += 1
                    if label_counter[label] > cap:
                        break
                    lines.append('{},{},{}\n'.format(os.path.basename(img), data_str, label))
            random.shuffle(lines)
            for line in lines:
                f.write(line)
        print('Done')




if __name__ == '__main__':

    for ocr_model in ["char2", "dict2"]:
        for traintest in ["train", "test"]:
            
            cds = CaffeDataset(srcfolder, ocr_model, traintest, outpath, permutate=False)
            cds.read_labels()
            cds.read_data()
            cds.write_caffe_data()

