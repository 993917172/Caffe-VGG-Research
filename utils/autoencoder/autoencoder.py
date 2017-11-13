from caffe import layers as L, params as P 
import sys
sys.path.append("utils") 
sys.path.append("utils/autoencoder") 
from basis import *
from __future__ import print_function

height=None
width=None

def conv1_autoencoder(split, batch_sz):
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(image_data_param=dict(source=split, batch_size=batch_sz,new_height=height, new_width=width,is_color=False),ntop=2)
    n.silence = L.Silence(n.label, ntop=0)
    n.flatdata_i = L.Flatten(n.data)
    
    n.conv1 = conv(n.data, 5, 5, 64, pad=2)
    n.bn1 = L.BatchNorm(n.conv1, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale1 = L.Scale(n.bn1, bias_term=True, in_place=True)    
    n.relu1 = L.ReLU(n.scale1, relu_param=dict(negative_slope=0.1))
    n.pool1 = max_pool(n.relu1, 2, stride=2)   
    
    n.code = conv(n.pool1, 5, 5, 64, pad=2)
    
    n.upsample1 = L.Deconvolution(n.code, param=dict(lr_mult=0 ,decay_mult = 0), convolution_param=dict(group=64, num_output=64, kernel_size=4, stride=2, pad=1, bias_term=False,weight_filler=dict(type="bilinear")))
    n.deconv1 = conv(n.upsample1, 5, 5, 1, pad=2)    
    n.debn1 = L.BatchNorm(n.deconv1, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.descale1 = L.Scale(n.debn1, bias_term=True, in_place=True) 
    n.derelu1 = L.ReLU(n.descale1, relu_param=dict(negative_slope=0.1))
    
    n.flatdata_o = L.Flatten(n.derelu1)
    n.loss_s = L.SigmoidCrossEntropyLoss(n.flatdata_o, n.flatdata_i, loss_weight=1)
    n.loss_e = L.EuclideanLoss(n.flatdata_o, n.flatdata_i, loss_weight=0)

    return str(n.to_proto())

def conv2_autoencoder(split, batch_sz):
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(image_data_param=dict(source=split, batch_size=batch_sz,new_height=height, new_width=width,is_color=False),ntop=2)
    n.silence = L.Silence(n.label, ntop=0)
    n.flatdata_i = L.Flatten(n.data)
    
    n.conv1 = conv(n.data, 5, 5, 64, pad=2, no_back=True)
    n.bn1 = L.BatchNorm(n.conv1, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale1 = L.Scale(n.bn1, bias_term=True, in_place=True)    
    n.relu1 = L.ReLU(n.scale1,relu_param=dict(negative_slope=0.1))
    n.pool1 = max_pool(n.relu1, 2, stride=2)  
    
    n.conv2 = conv(n.pool1, 5, 5, 128, pad=2)
    n.bn2 = L.BatchNorm(n.conv2, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale2 = L.Scale(n.bn2, bias_term=True, in_place=True) 
    n.relu2 = L.ReLU(n.scale2, relu_param=dict(negative_slope=0.1))
    n.pool2 = max_pool(n.relu2, 2, stride=2)
    
    n.code = conv(n.pool2, 5, 5, 128, pad=2)

    n.upsample2 = L.Deconvolution(n.code, param=dict(lr_mult=0 ,decay_mult = 0), convolution_param=dict(group=128, num_output=128, kernel_size=4, stride=2, pad=1, bias_term=False,weight_filler=dict(type="bilinear")))
    n.deconv2 = conv(n.upsample2, 5, 5, 64, pad=2) 
    n.debn2 = L.BatchNorm(n.deconv2, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.descale2 = L.Scale(n.debn2, bias_term=True, in_place=True) 
    n.derelu2 = L.ReLU(n.descale2, relu_param=dict(negative_slope=0.1))

    n.upsample1 = L.Deconvolution(n.derelu2, param=dict(lr_mult=0 ,decay_mult = 0), convolution_param=dict(group=64, num_output=64, kernel_size=4, stride=2, pad=1, bias_term=False,weight_filler=dict(type="bilinear")))
    n.deconv1 = conv(n.upsample1, 5, 5, 1, pad=2, no_back=True)    
    n.debn1 = L.BatchNorm(n.deconv1, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.descale1 = L.Scale(n.debn1, bias_term=True, in_place=True) 
    n.derelu1 = L.ReLU(n.descale1, relu_param=dict(negative_slope=0.1))

    n.flatdata_o = L.Flatten(n.derelu1)
    n.loss_s = L.SigmoidCrossEntropyLoss(n.flatdata_o, n.flatdata_i, loss_weight=1)
    n.loss_e = L.EuclideanLoss(n.flatdata_o, n.flatdata_i, loss_weight=0)
    
    
    return str(n.to_proto())

def conv3_autoencoder(split, batch_sz):
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(image_data_param=dict(source=split, batch_size=batch_sz,new_height=height, new_width=width, is_color=False),ntop=2)
    n.silence = L.Silence(n.label, ntop=0)
    n.flatdata_i = L.Flatten(n.data)
    
    n.conv1 = conv(n.data, 5, 5, 64, pad=2, no_back=True)
    n.bn1 = L.BatchNorm(n.conv1, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale1 = L.Scale(n.bn1, bias_term=True, in_place=True)    
    n.relu1 = L.ReLU(n.scale1, relu_param=dict(negative_slope=0.1))
    n.pool1 = max_pool(n.relu1, 2, stride=2)  
    
    n.conv2 = conv(n.pool1, 5, 5, 128, pad=2, no_back=True)
    n.bn2 = L.BatchNorm(n.conv2, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale2 = L.Scale(n.bn2, bias_term=True, in_place=True) 
    n.relu2 = L.ReLU(n.scale2, relu_param=dict(negative_slope=0.1))
    n.pool2 = max_pool(n.relu2, 2, stride=2)
    
    n.conv3 = conv(n.pool2, 3, 3, 256, pad=1)
    n.bn3 = L.BatchNorm(n.conv3, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale3 = L.Scale(n.bn3, bias_term=True, in_place=True)  
    n.relu3 = L.ReLU(n.scale3, relu_param=dict(negative_slope=0.1))    
    n.conv3_5 = conv(n.relu3, 3, 3, 512, pad=1)
    n.bn3_5 = L.BatchNorm(n.conv3_5, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale3_5 = L.Scale(n.bn3_5, bias_term=True, in_place=True)  
    n.relu3_5 = L.ReLU(n.scale3_5, relu_param=dict(negative_slope=0.1))
    n.pool3_5 = max_pool(n.relu3_5, 2, stride=2)
    
    n.code = conv(n.pool3_5, 3, 3, 512, pad=1)


    n.upsample3_5 = L.Deconvolution(n.code, param=dict(lr_mult=0 ,decay_mult = 0), convolution_param=dict(group=512, num_output=512, kernel_size=4, stride=2, pad=1, bias_term=False,weight_filler=dict(type="bilinear")))
    n.deconv3_5 = conv(n.upsample3_5, 3, 3, 256, pad=1, no_back=True)    
    n.debn3_5 = L.BatchNorm(n.deconv3_5, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.descale3_5 = L.Scale(n.debn3_5, bias_term=True, in_place=True) 
    n.derelu3_5 = L.ReLU(n.descale3_5, relu_param=dict(negative_slope=0.1))
                                  
    n.deconv3 = conv(n.derelu3_5, 5, 5,128, pad=2, no_back=True)    
    n.debn3 = L.BatchNorm(n.deconv3, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.descale3 = L.Scale(n.debn3, bias_term=True, in_place=True) 
    n.derelu3 = L.ReLU(n.descale3, relu_param=dict(negative_slope=0.1))
    
    n.upsample2 = L.Deconvolution(n.derelu3, param=dict(lr_mult=0 ,decay_mult = 0), convolution_param=dict(group=128, num_output=128, kernel_size=4, stride=2, pad=1, bias_term=False,weight_filler=dict(type="bilinear")))
    n.deconv2 = conv(n.upsample2, 5, 5, 64, pad=2, no_back=True) 
    n.debn2 = L.BatchNorm(n.deconv2, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.descale2 = L.Scale(n.debn2, bias_term=True, in_place=True) 
    n.derelu2 = L.ReLU(n.descale2, relu_param=dict(negative_slope=0.1))

    n.upsample1 = L.Deconvolution(n.derelu2, param=dict(lr_mult=0 ,decay_mult = 0), convolution_param=dict(group=64, num_output=64, kernel_size=4, stride=2, pad=1, bias_term=False,weight_filler=dict(type="bilinear")))
    n.deconv1 = conv(n.upsample1, 5, 5, 1, pad=2, no_back=True)    
    n.debn1 = L.BatchNorm(n.deconv1, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.descale1 = L.Scale(n.debn1, bias_term=True, in_place=True) 
    n.derelu1 = L.ReLU(n.descale1, relu_param=dict(negative_slope=0.1))

    n.flatdata_o = L.Flatten(n.derelu1)
    n.loss_s = L.SigmoidCrossEntropyLoss(n.flatdata_o, n.flatdata_i, loss_weight=1)
    n.loss_e = L.EuclideanLoss(n.flatdata_o, n.flatdata_i, loss_weight=0)
    
    
    return str(n.to_proto())


def conv4_autoencoder(split, batch_sz):
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(image_data_param=dict(source=split, batch_size=batch_sz,new_height=height, new_width=width,is_color=False),ntop=2)
    n.silence = L.Silence(n.label, ntop=0)
    n.flatdata_i = L.Flatten(n.data)
    
    n.conv1 = conv(n.data, 5, 5, 64, pad=2, no_back=True)
    n.bn1 = L.BatchNorm(n.conv1, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale1 = L.Scale(n.bn1, bias_term=True, in_place=True)    
    n.relu1 = L.ReLU(n.scale1, relu_param=dict(negative_slope=0.1))
    n.pool1 = max_pool(n.relu1, 2, stride=2)  
    
    n.conv2 = conv(n.pool1, 5, 5, 128, pad=2, no_back=True)
    n.bn2 = L.BatchNorm(n.conv2, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale2 = L.Scale(n.bn2, bias_term=True, in_place=True) 
    n.relu2 = L.ReLU(n.scale2, relu_param=dict(negative_slope=0.1))
    n.pool2 = max_pool(n.relu2, 2, stride=2)
    
    n.conv3 = conv(n.pool2, 3, 3, 256, pad=1, no_back=True)
    n.bn3 = L.BatchNorm(n.conv3, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale3 = L.Scale(n.bn3, bias_term=True, in_place=True)  
    n.relu3 = L.ReLU(n.scale3, relu_param=dict(negative_slope=0.1))    
    n.conv3_5 = conv(n.relu3, 3, 3, 512, pad=1, no_back=True)
    n.bn3_5 = L.BatchNorm(n.conv3_5, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale3_5 = L.Scale(n.bn3_5, bias_term=True, in_place=True)  
    n.relu3_5 = L.ReLU(n.scale3_5, relu_param=dict(negative_slope=0.1))
    n.pool3_5 = max_pool(n.relu3_5, 2, stride=2)
    
    n.conv4 = conv(n.pool3_5, 3, 3, 512, pad=1)
    n.bn4 = L.BatchNorm(n.conv4, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale4 = L.Scale(n.bn4, bias_term=True, in_place=True)  
    n.relu4 = L.ReLU(n.scale4, relu_param=dict(negative_slope=0.1))
                                  
    n.code = conv(n.relu4, 3, 3, 512, pad=1)
    
    n.deconv4 = conv(n.code, 3, 3, 512, pad=1)    
    n.debn4 = L.BatchNorm(n.deconv4, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.descale4 = L.Scale(n.debn4, bias_term=True, in_place=True) 
    n.derelu4 = L.ReLU(n.descale4, relu_param=dict(negative_slope=0.1))                                      

    n.upsample3_5 = L.Deconvolution(n.derelu4, param=dict(lr_mult=0 ,decay_mult = 0), convolution_param=dict(group=512, num_output=512, kernel_size=4, stride=2, pad=1, bias_term=False,weight_filler=dict(type="bilinear")))
    n.deconv3_5 = conv(n.upsample3_5, 3, 3, 256, pad=1, no_back=True)    
    n.debn3_5 = L.BatchNorm(n.deconv3_5, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.descale3_5 = L.Scale(n.debn3_5, bias_term=True, in_place=True) 
    n.derelu3_5 = L.ReLU(n.descale3_5, relu_param=dict(negative_slope=0.1))
                                  
    n.deconv3 = conv(n.derelu3_5, 5, 5,128, pad=2, no_back=True)    
    n.debn3 = L.BatchNorm(n.deconv3, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.descale3 = L.Scale(n.debn3, bias_term=True, in_place=True) 
    n.derelu3 = L.ReLU(n.descale3, relu_param=dict(negative_slope=0.1))
    
    n.upsample2 = L.Deconvolution(n.derelu3, param=dict(lr_mult=0 ,decay_mult = 0), convolution_param=dict(group=128, num_output=128, kernel_size=4, stride=2, pad=1, bias_term=False,weight_filler=dict(type="bilinear")))
    n.deconv2 = conv(n.upsample2, 5, 5, 64, pad=2, no_back=True) 
    n.debn2 = L.BatchNorm(n.deconv2, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.descale2 = L.Scale(n.debn2, bias_term=True, in_place=True) 
    n.derelu2 = L.ReLU(n.descale2, relu_param=dict(negative_slope=0.1))

    n.upsample1 = L.Deconvolution(n.derelu2, param=dict(lr_mult=0 ,decay_mult = 0), convolution_param=dict(group=64, num_output=64, kernel_size=4, stride=2, pad=1, bias_term=False,weight_filler=dict(type="bilinear")))
    n.deconv1 = conv(n.upsample1, 5, 5, 1, pad=2, no_back=True)    
    n.debn1 = L.BatchNorm(n.deconv1, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.descale1 = L.Scale(n.debn1, bias_term=True, in_place=True) 
    n.derelu1 = L.ReLU(n.descale1, relu_param=dict(negative_slope=0.1))

    n.flatdata_o = L.Flatten(n.derelu1)
    n.loss_s = L.SigmoidCrossEntropyLoss(n.flatdata_o, n.flatdata_i, loss_weight=1)
    n.loss_e = L.EuclideanLoss(n.flatdata_o, n.flatdata_i, loss_weight=0)
    return str(n.to_proto())


def vgg(split, batch_sz):
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(image_data_param=dict(shuffle=True,source=split, batch_size=batch_sz,new_height=32, new_width=100,is_color=False),ntop=2)
    n.silence = L.Silence(n.label, ntop=0)
    
    n.conv1 = conv(n.data, 5, 5, 64, pad=2)
    n.bn1 = L.BatchNorm(n.conv1, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale1 = L.Scale(n.bn1, bias_term=True, in_place=True)    
    n.relu1 = L.ReLU(n.scale1)
    n.pool1 = max_pool(n.relu1, 2, stride=2)  
    
    n.conv2 = conv(n.pool1, 5, 5, 128, pad=2)
    n.bn2 = L.BatchNorm(n.conv2, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale2 = L.Scale(n.bn2, bias_term=True, in_place=True) 
    n.relu2 = L.ReLU(n.scale2)
    n.pool2 = max_pool(n.relu2, 2, stride=2)
    
    n.conv3 = conv(n.pool2, 3, 3, 256, pad=1)
    n.bn3 = L.BatchNorm(n.conv3, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale3 = L.Scale(n.bn3, bias_term=True, in_place=True)  
    n.relu3 = L.ReLU(n.scale3)    
    n.conv3_5 = conv(n.relu3, 3, 3, 512, pad=1)
    n.bn3_5 = L.BatchNorm(n.conv3_5, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale3_5 = L.Scale(n.bn3_5, bias_term=True, in_place=True)  
    n.relu3_5 = L.ReLU(n.scale3_5)
    n.pool3_5 = max_pool(n.relu3_5, 2, stride=2)
    
    n.conv4 = conv(n.pool3_5, 3, 3, 512, pad=1)
    n.bn4 = L.BatchNorm(n.conv4, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale4 = L.Scale(n.bn4, bias_term=True, in_place=True)  
    n.relu4 = L.ReLU(n.scale4)
    
    n.fc5 = conv(n.relu4, 13, 4, 4096)
    n.bn5 = L.BatchNorm(n.fc1, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale5 = L.Scale(n.bn5, bias_term=True, in_place=True)  
    n.relu5 = L.ReLU(n.scale5)
    n.drop1 = L.Dropout(n.relu5, in_place=True)
    
    n.fc6 = conv(n.drop1, 1, 1, 4096)
    n.bn6 = L.BatchNorm(n.fc2, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale6 = L.Scale(n.bn6, bias_term=True, in_place=True)  
    n.relu6 = L.ReLU(n.scale6)
    n.drop2 = L.Dropout(n.relu6, in_place=True)
    
    n.fc_class = conv(n.drop2, 1, 1, 88172)
    n.bn7 = L.BatchNorm(n.fc_class, use_global_stats=False, in_place=True, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.scale7 = L.Scale(n.bn7, bias_term=True, in_place=True)  
    n.relu7 = L.ReLU(n.scale7) 

    n.loss = L.SoftmaxWithLoss(n.relu7, n.label, loss_weight=1)
    return str(n.to_proto())


