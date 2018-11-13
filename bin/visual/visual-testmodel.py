import sys
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

caffe_root = '../../caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
weights_path="../results/visual/"
test="../dataset/test.txt"


if __name__ == '__main__':
    if len(sys.argv) < 3 :
        print('Usage: {} CNN-name (VGG16-AlexNet-CaffeNet-GoogLeNet-ResNet50-ResNet101-ResNet152)'.format(sys.argv[0]))
        sys.exit()
    cnn = sys.argv[1]
    n_iter=int(sys.argv[2])
    if cnn not in ["VGG16","AlexNet","CaffeNet","GoogLeNet","ResNet50","ResNet101","ResNet152"]:
        print('Usage: {} CNN-name (VGG16-AlexNet-CaffeNet-GoogLeNet-ResNet50-ResNet101-ResNet152)'.format(sys.argv[0]))
        sys.exit()
    else:

        if cnn=="VGG16":
            model_def = caffe_root + 'models/VGG16/retail/deploy.prototxt'
            model_weights = weights_path + 'vgg16-snapshot_iter_{}.caffemodel'.format(n_iter)
        elif cnn=="AlexNet":
            model_def = caffe_root + 'models/bvlc_alexnet/retail/deploy.prototxt'
            model_weights = weights_path + 'alexnet-snapshot_iter_{}.caffemodel'.format(n_iter)
        elif cnn=="CaffeNet":
            model_def = caffe_root + 'models/bvlc_reference_caffenet/retail/deploy.prototxt'
            model_weights = weights_path + 'caffenet-snapshot_iter_{}.caffemodel'.format(n_iter)
        elif cnn=="GoogLeNet":
            model_def = caffe_root + 'models/bvlc_googlenet/retail/deploy.prototxt'
            model_weights = weights_path + 'googlenet-snapshot_iter_{}.caffemodel'.format(n_iter)
        elif cnn=="ResNet50":
            model_def = caffe_root + 'models/resnet50/retail/deploy.prototxt'
            model_weights = weights_path + 'resnet50-snapshot_iter_{}.caffemodel'.format(n_iter)
        elif cnn=="ResNet101":
            model_def = caffe_root + 'models/resnet101/retail/deploy.prototxt'
            model_weights = weights_path + 'resnet101-snapshot_iter_{}.caffemodel'.format(n_iter)
        elif cnn=="ResNet152":
            model_def = caffe_root + 'models/resnet152/retail/deploy.prototxt'
            model_weights = weights_path + 'resnet152-snapshot_iter_{}.caffemodel'.format(n_iter)


    print("Caffe Loading")
    sys.path.insert(0, caffe_root + 'python')
    import caffe

    caffe.set_mode_gpu()

    print("Network Loading: "+cnn)
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)
    print("Network Mean Loading")
    import numpy as np
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    if cnn in ["ResNet50","ResNet101","ResNet152"]:
        mu = np.load(caffe_root + 'models/resnet50/ResNet_mean.npy')
    else:
        mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    print 'mean-subtracted values:', zip('BGR', mu)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    # set the size of the input (we can skip this if we're happy
    #  with the default; we can also change it later, e.g., for different batch sizes)
    net.blobs['data'].reshape(  1,        # batch size
                                3,         # 3-channel (BGR) images
                                224, 224)  # image size is 224x224



    predicted_lables=[]
    true_labels = []
    class_names = ['SOOS','Normal','Promotion']
    print("TEST...")
    with open(test, 'rt') as f:
        for line in [line.strip() for line in f.readlines()]:
            cols = [col.strip() for col in line.split(' ', 3)]
            img, sent = cols

            # transform it and copy it into the net
            image = caffe.io.load_image(img)
            net.blobs['data'].data[...] = transformer.preprocess('data', image)

            # perform classification
            net.forward()

            # obtain the output probabilities
            output_prob = net.blobs['softmax'].data[0]

            plabel = int(output_prob.argmax())
            predicted_lables.append(plabel)	
            true_labels.append(int(sent))
        
        print( confusion_matrix(y_true=true_labels, y_pred=predicted_lables, labels=[0,1,2]))
        print( classification_report(y_true=true_labels, y_pred=predicted_lables, target_names=class_names))
    print("TEST Ended!!")
