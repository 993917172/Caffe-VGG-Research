import sys
caffe_root = '../../../../'  # this file is expected to be in {caffe_root}/examples
ml_root = '../../'
sys.path.append(caffe_root + 'python')
import caffe # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
from caffe import layers as L, params as P # Shortcuts to define the net prototxt.

sys.path.append(ml_root + "pycaffe/layers") # the datalayers we will use are in this directory.
sys.path.append(ml_root + "pycaffe") # the tools file is in this folder

# helper function for common structures
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, group=group, weight_filler=dict(type='gaussian',std=0.01), bias_filler=dict(type='constant', value=0))
    
    return conv, L.ReLU(conv, in_place=True)

def fc_conv(bottom,  nout, stride=1, pad=0, group=1, ks=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, group=group, weight_filler=dict(type='gaussian',std=0.01), bias_filler=dict(type='constant', value=0))
    return conv


def fc_conv_relu(bottom, k_w, k_h, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_w=k_w, kernel_h=k_h, stride=stride, num_output=nout, pad=pad, group=group, weight_filler=dict(type='gaussian',std=0.01), bias_filler=dict(type='constant', value=0))

       
    return conv, L.ReLU(conv, in_place=True)

# another helper function
def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

def in_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

# yet another helper function
def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

