import sys
caffe_root = '../../../../'  # this file is expected to be in {caffe_root}/examples
ml_root = '../../'
import os
import os.path as osp
import numpy as np
sys.path.append(caffe_root + 'python')
import caffe # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
from caffe import layers as L, params as P # Shortcuts to define the net prototxt.

sys.path.append(ml_root + "pycaffe/layers") # the datalayers we will use are in this directory.
sys.path.append(ml_root + "pycaffe") # the tools file is in this folder
sys.path.append(ml_root + "pycaffe/net_shrink")
from nets import *
import heapq
import tools_gray as tools#this contains some tools that we need


from basis import *

def train_and_test(pro, source_folder, working_folder, num, class_num):

    postfix = "_fc1_%d_fc2_%d.prototxt" % (num, num)
    pro.make_prototxt(postfix, [4096-num, 4096-num, class_num], str(num))
    solver = caffe.SGDSolver(osp.join(working_folder, "prototxt", str(num), "solver"+postfix))
    solver.net.copy_from(osp.join(source_folder,"snap_fc1_%d_fc2_%d.caffemodel"%(num, num)))
    solver.test_nets[0].share_with(solver.net)

    return solver
