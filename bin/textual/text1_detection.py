from __future__ import print_function
import sys
import os
import numpy as np
from PIL import Image
import datetime

dataset_path="/home/massimo/Scrivania/tirocinio/retail/dataset/RETAIL-Original/"

# Make sure that custom caffe is on the python path:
caffe_root = '../../txt/TextBoxes'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe


caffe.set_mode_cpu()
print("Network Loading")
model_def = '../../txt/TextBoxes/examples/TextBoxes/deploy.prototxt'
model_weights = '../../txt/TextBoxes/TextBoxes_icdar13.caffemodel'
scales = ((700, 700),)
net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
print(net.blobs['data'].data.shape)
    
for folder in ["train","test"]:
    
    now = datetime.datetime.now()
    print (folder+" - Current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))
    
    input_txt = "../dataset/"+folder+".txt"         #path for train.txt and test.txt
    dest_path = "../results/textual/"+folder+"1"    #path for results

        
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    imgs = []

    # load image list
    with open(input_txt, 'rt') as f:
        for line in [line.strip() for line in f.readlines()]:
            cols = [col.strip() for col in line.split(' ')]
            img_path= cols[0]
            #img = cols[1]
            #img_path = os.path.join(img_root, subfolder, img)
            imgs.append(img_path)

    #processed images in previous elaborations
    txtdone="../results/textual/"+folder+"1-done.txt"
    imgsdone= []
    if os.path.exists(txtdone):
        with open(txtdone, 'rt') as f:
            for line in [line.strip() for line in f.readlines()]:
                imgsdone.append(line)
    savedone = open(txtdone, 'at')
    
    cont=1
    for image_path in imgs:
        img_base = os.path.basename(image_path)
        print('Processing image {} of {} - {}'.format(cont,len(imgs),img_base))

        if img_base in imgsdone:
            print("{} skipped...".format(img_base))
            cont+=1
            continue

        save_detection_path = os.path.join(dest_path, img_base + '.crop_coords.txt')
        detection_result = open(save_detection_path, 'wt')
        

        cv_img = Image.open(dataset_path+img_base)
        for scale in scales:
            image_resize_height = scale[0]
            image_resize_width = scale[1]
            transformer = caffe.io.Transformer({'data': (1, 3, image_resize_height, image_resize_width)})
            transformer.set_transpose('data', (2, 0, 1))
            transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
            transformer.set_raw_scale('data',
                                      255)  # the reference model operates on images in [0,255] range instead of [0,1]
            transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

            net.blobs['data'].reshape(1, 3, image_resize_height, image_resize_width)
            transformed_image = transformer.preprocess('data', image)
            net.blobs['data'].data[...] = transformed_image
            # Forward pass.
            detections = net.forward()['detection_out']

            # Parse the outputs.
            det_label = detections[0, 0, :, 1]
            det_conf = detections[0, 0, :, 2]
            det_xmin = detections[0, 0, :, 3]
            det_ymin = detections[0, 0, :, 4]
            det_xmax = detections[0, 0, :, 5]
            det_ymax = detections[0, 0, :, 6]

            top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

            top_conf = det_conf[top_indices]
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]

            for i in xrange(top_conf.shape[0]):
                xmin = int(round(top_xmin[i] * image.shape[1]))
                ymin = int(round(top_ymin[i] * image.shape[0]))
                xmax = int(round(top_xmax[i] * image.shape[1]))
                ymax = int(round(top_ymax[i] * image.shape[0]))
                xmin = max(1, xmin)
                ymin = max(1, ymin)
                xmax = min(image.shape[1] - 1, xmax)
                ymax = min(image.shape[0] - 1, ymax)
                score = top_conf[i]
                coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
                score = top_conf[i]
                coord_string = ','.join([str(coord) for coord in [xmin, ymin, xmax, ymax]])
                crop_filn = os.path.join(dest_path, img_base + '.crop.' + str(i) + '.png')
                detection_result.write('{},{}\n'.format(os.path.basename(crop_filn), coord_string))
                crop = cv_img.crop((xmin, ymin, xmax, ymax))
                crop.save(crop_filn)

        detection_result.close()
        savedone.write('{}\n'.format(img_base))
        cont+=1

    savedone.close()
    print(folder+' success!')
    
now = datetime.datetime.now()
print ("Current date and time : ")
print (now.strftime("%Y-%m-%d %H:%M:%S"))