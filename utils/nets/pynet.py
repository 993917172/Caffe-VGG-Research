import sys
caffe_root = '../../../'
utils_root = '../'

sys.path.append(caffe_root + 'python')

import caffe 
from caffe import layers as L, params as P # Shortcuts to define the net prototxt.

sys.path.append(utils_root + "layers") 
sys.path.append(utils_root + "nets") 

from basis import *
from multilabel_datalayers import *

def multilabel_bvlc(data_layer_params,num_data):

    n = caffe.NetSpec()
    n.data, n.label = L.Python(module = 'multilabel_datalayers', layer = 'MultilabelDataLayerSync',
                               ntop = 2, param_str=str(data_layer_params))

    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096)
    n.drop6 = L.Dropout(n.relu6, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.drop6, 4096)
    n.drop7 = L.Dropout(n.relu7, in_place=True)
    n.score = L.InnerProduct(n.drop7, num_output=num_data)
    n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)
    
    return str(n.to_proto())

def multilabel_vgg_dictnet(data_layer_params, num_data):

    n = caffe.NetSpec()
    n.data, n.label = L.Python(module = 'multilabel_datalayers', layer = 'MultilabelDataLayerSync', 
                               ntop = 2, param_str=str(data_layer_params))

    n.conv1, n.relu1 = conv_relu(n.data, 5, 64, pad=2)
    n.pool1 = max_pool(n.relu1, 2, stride=2)
    n.conv2, n.relu2 = conv_relu(n.pool1, 5, 128, pad=2)
    n.pool2 = max_pool(n.relu2, 2, stride=2)
    n.conv3, n.relu3 = conv_relu(n.pool2, 3, 256, pad=1)
    n.conv3_5, n.relu3_5 = conv_relu(n.relu3, 3, 512, pad=1)
    n.pool3 = max_pool(n.relu3_5, 2, stride=2)
    n.conv4, n.relu4 = conv_relu(n.pool3, 3, 512, pad=1)    

    n.fc1, n.relu5 = fc_conv_relu(n.relu4,13,4, 4096)
    n.drop6 = L.Dropout(n.relu5, in_place=True)    
    n.fc2, n.relu6 = fc_conv_relu(n.drop6,1,1, 4096)
    n.drop7 = L.Dropout(n.relu6, in_place=True)    
    n.score = fc_conv(n.drop7, num_data)
    n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)
    
    return str(n.to_proto())

def multilabel_vgg16(data_layer_params, num_data):
    # setup the python data layer 
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module = 'multilabel_datalayers', layer = 'MultilabelDataLayerSync', 
                               ntop = 2, param_str=str(data_layer_params))   
    
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 3, 64, pad=1)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 3, 64)
    n.pool1 = max_pool(n.relu1_2, 2, stride=2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 3, 128, pad=1)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 3, 128, pad=1)
    n.pool2 = max_pool(n.relu2_2, 2, stride=2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 3, 256, pad=1)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 3, 256, pad=1)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 3, 256, pad=1)
    n.pool3 = max_pool(n.relu3_3, 2, stride=2)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 3, 512, pad=1)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 3, 512, pad=1)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 3, 512, pad=1)
    n.pool4 = max_pool(n.relu4_3, 2, stride=2)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 3, 512, pad=1)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 3, 512, pad=1)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 3, 512, pad=1)
    n.pool5 = max_pool(n.relu5_3, 2, stride=2)    
    
    n.fc6, n.relu6 = in_relu(n.pool5, 4096)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = in_relu(n.drop6, 4096)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    n.out = L.InnerProduct(n.drop7, num_output=num_data)
    n.loss = L.SigmoidCrossEntropyLoss(n.out, n.label)
    
       
    return str(n.to_proto())


def multilabel_large_vgg(data_layer_params, num_data):
    # setup the python data layer 
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module = 'multilabel_datalayers', layer = 'MultilabelDataLayerSync', 
                               ntop = 2, param_str=str(data_layer_params))

    n.conv1, n.relu1 = conv_relu(n.data, 5, 64, pad=2)
    n.pool1 = max_pool(n.relu1, 2, stride=2)
    n.conv2, n.relu2 = conv_relu(n.pool1, 5, 128, pad=2)
    n.pool2 = max_pool(n.relu2, 2, stride=2)
    n.conv3, n.relu3 = conv_relu(n.pool2, 3, 256, pad=1)
    n.conv3_5, n.relu3_5 = conv_relu(n.relu3, 3, 512, pad=1)
    n.pool3 = max_pool(n.relu3_5, 2, stride=2)
    
    n.conv4, n.relu4 = conv_relu(n.pool3, 3, 512, pad=1) 

    n.fc1, n.relu5 = fc_conv_relu(n.relu4,30,8, 4096)
    
    n.drop6 = L.Dropout(n.relu5, in_place=True)    
    n.fc2, n.relu6 = fc_conv_relu(n.drop6,1,1, 4096)
    n.drop8 = L.Dropout(n.relu6, in_place=True)    
    
    n.score = fc_conv(n.drop8, num_data)
    n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)
    
    return str(n.to_proto())
