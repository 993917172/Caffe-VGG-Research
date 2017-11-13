import sys
caffe_root = '../../../../'  # this file is expected to be in {caffe_root}/examples
ml_root = '../../'
sys.path.append(caffe_root + 'python')
import caffe # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
from caffe import layers as L, params as P # Shortcuts to define the net prototxt.

sys.path.append(ml_root + "pycaffe/layers") # the datalayers we will use are in this directory.
sys.path.append(ml_root + "pycaffe") # the tools file is in this folder

from basis import *


def caffenet_multilabel_vgg(split, batch_sz, fc1_kernel, fc_wei_num, is_training=True):
    # setup the python data layer 
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(image_data_param=dict(source=split, batch_size=batch_sz,new_height=32, new_width=100,is_color=False),ntop=2)

    # the net itself
    n.conv1, n.relu1 = conv_relu(n.data, 5, 64, pad=2)
    n.pool1 = max_pool(n.relu1, 2, stride=2)
    n.conv2, n.relu2 = conv_relu(n.pool1, 5, 128, pad=2)
    n.pool2 = max_pool(n.relu2, 2, stride=2)
    n.conv3, n.relu3 = conv_relu(n.pool2, 3, 256, pad=1)
    n.conv3_5, n.relu3_5 = conv_relu(n.relu3, 3, 512, pad=1)
    n.pool3 = max_pool(n.relu3_5, 2, stride=2)
    n.conv4, n.relu4 = conv_relu(n.pool3, 3, 512, pad=1)    
    n.fc1, n.relu5 = fc_conv_relu(n.relu4, fc1_kernel[1], fc1_kernel[0], fc_wei_num[0])
    #n.drop6 = L.Dropout(n.relu5, in_place=True)    
    n.fc2, n.relu6 = fc_conv_relu(n.fc1,1,1, fc_wei_num[1])
    #n.drop7 = L.Dropout(n.relu6, in_place=True)    
    n.fc_class = fc_conv(n.fc2, fc_wei_num[2])
    if is_training is True:
        n.loss = L.SoftmaxWithLoss(n.fc_class, n.label)
#    n.loss = L.SoftmaxWithLoss(n.score, n.label)
    
    return str(n.to_proto())


def caffenet_vgg_input(batch_sz, fc1_kernel, fc_wei_num, is_training=True):
    # setup the python data layer 
    n = caffe.NetSpec()
    n.data = L.Input(shape=dict(dim=[batch_sz, 1, 32, 100]))

    # the net itself
    n.conv1, n.relu1 = conv_relu(n.data, 5, 64, pad=2)
    n.pool1 = max_pool(n.relu1, 2, stride=2)
    n.conv2, n.relu2 = conv_relu(n.pool1, 5, 128, pad=2)
    n.pool2 = max_pool(n.relu2, 2, stride=2)
    n.conv3, n.relu3 = conv_relu(n.pool2, 3, 256, pad=1)
    n.conv3_5, n.relu3_5 = conv_relu(n.relu3, 3, 512, pad=1)
    n.pool3 = max_pool(n.relu3_5, 2, stride=2)
    n.conv4, n.relu4 = conv_relu(n.pool3, 3, 512, pad=1)    
    n.fc1, n.relu5 = fc_conv_relu(n.relu4, fc1_kernel[1], fc1_kernel[0], fc_wei_num[0])
    n.fc2, n.relu6 = fc_conv_relu(n.relu5,1,1, fc_wei_num[1])
    n.fc_class = fc_conv(n.relu6, fc_wei_num[2])
    n.prob = L.Softmax(n.fc_class)
    
    return str(n.to_proto())
