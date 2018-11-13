#!/usr/bin/env python
from __future__ import print_function
import sys
import os
from collections import defaultdict

base_folder="../results/textual/"

if __name__ == '__main__':
    for tipo in ["train","test"]:
        folder=base_folder+tipo
        if not os.path.exists(folder+"4"):
        	os.makedirs(folder+"4")
        print(folder)
        # find all crop_in_order.txts
        cont=1
        for croplist_filn in [f for f in os.listdir(folder+"2") if f.endswith('crops_in_order.txt')]:
            croplist_filp = os.path.join(folder+"2", croplist_filn)
            print('{} - Processing'.format(cont,croplist_filn))
            cont+=1
            with open (croplist_filp, 'rt') as f:
    			# find out crop order
    			recog_text = defaultdict(str)
    			for line in [line.strip() for line in f.readlines()]:
    				cropname = os.path.join(folder+"3", line.split(',')[0])
    				for model in 'char2', 'dict2':
    				    crop_text_fn = cropname + '.{}.txt'.format(model)
    				    if os.path.exists(crop_text_fn):
    				        with open(crop_text_fn, 'rt') as tf:
    				            print('    ', crop_text_fn)
    				            text = tf.readline().strip()
    				            recog_text[model] += '{} '.format(text)
    	    for model in recog_text:
    			out_filn = croplist_filp.replace('.crops_in_order.txt', '.all.{}.txt'.format(model))	
        		out2=folder+"4/"+os.path.basename(out_filn)
    			with open(out2, 'wt') as f:
    				f.write('{}\n'.format(recog_text[model]))
