import sys
import os

caffe_root = '../../caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
weights_path="../results/visual/"
traintest_path="../dataset/"


if __name__ == '__main__':
    if len(sys.argv) < 2 :
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


    base=weights_path+"img_features/"+cnn+"/"
    if not os.path.exists(base):
        os.makedirs(base)
    for fase in ["train","test"]:
        imgfile=traintest_path+fase+".txt"
        
        if cnn=="GoogLeNet":
            print("FEATURES EXTRACTION - Network: "+cnn+" - layers: inception_5a, inception_5b, pool5 - phase: "+fase+"...")
            outputfile1 = base+fase+"_inception_5a.features.csv"
            outputfile2 = base+fase+"_inception_5b.features.csv"
            outputfile3 = base+fase+"_pool5.features.csv"
        elif cnn in ["ResNet50","ResNet101","ResNet152"]:
            print("FEATURES EXTRACTION - Network: "+cnn+" - layers: res5b, res5c, pool5 - phase: "+fase+"...")
            outputfile1 = base+fase+"_res5b.features.csv"
            outputfile2 = base+fase+"_res5c.features.csv"
            outputfile3 = base+fase+"_pool5.features.csv"
        else:
            print("FEATURES EXTRACTION - Network: "+cnn+" - layers: pool5, fc6, fc7 - phase: "+fase+"...")
            outputfile1 = base+fase+"_pool5.features.csv"
            outputfile2 = base+fase+"_fc6.features.csv"
            outputfile3 = base+fase+"_fc7.features.csv"
        with open(imgfile, 'rt') as f:
            with open(outputfile1, 'w') as writer1:
                with open(outputfile2, 'w') as writer2:
                    with open(outputfile3, 'w') as writer3:
                        writer1.truncate()
                        writer2.truncate()
                        writer3.truncate()
                        num=1
                        for line in [line.strip() for line in f.readlines()]:
                            if num%200==0:
                                print(" {}".format(num))
                            num=num+1
                            cols = [col.strip() for col in line.split(' ', 3)]
                            img, sent = cols
                        
                            # transform it and copy it into the net
                            image = caffe.io.load_image(img)
                            net.blobs['data'].data[...] = transformer.preprocess('data', image)

                            # perform classification
                            net.forward()

                            # obtain the output probabilities
                            #output_prob = net.blobs['softmax'].data[0]
                            if cnn=="GoogLeNet":
                                np.savetxt(writer1, net.blobs["inception_5a/pool"].data[0].reshape(1,-1), fmt='%.8g', delimiter=',')
                                np.savetxt(writer2, net.blobs["inception_5b/pool"].data[0].reshape(1,-1), fmt='%.8g', delimiter=',')
                                np.savetxt(writer3, net.blobs["pool5/7x7_s1"].data[0].reshape(1,-1), fmt='%.8g', delimiter=',')
                            elif cnn in ["ResNet50","ResNet101","ResNet152"]:
                                np.savetxt(writer1, net.blobs["res5b"].data[0].reshape(1,-1), fmt='%.8g', delimiter=',')
                                np.savetxt(writer2, net.blobs["res5c"].data[0].reshape(1,-1), fmt='%.8g', delimiter=',')
                                np.savetxt(writer3, net.blobs["pool5"].data[0].reshape(1,-1), fmt='%.8g', delimiter=',')
                            else:
                                np.savetxt(writer1, net.blobs["pool5"].data[0].reshape(1,-1), fmt='%.8g', delimiter=',')
                                np.savetxt(writer2, net.blobs["fc6"].data[0].reshape(1,-1), fmt='%.8g', delimiter=',')
                                np.savetxt(writer3, net.blobs["fc7"].data[0].reshape(1,-1), fmt='%.8g', delimiter=',')


        print("Phase Ended!!")
print("FEATURES EXTRACTION Ended!!")

