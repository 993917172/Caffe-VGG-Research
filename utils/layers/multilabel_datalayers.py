# imports
import scipy.misc
import caffe
import numpy as np
import os.path as osp
from random import shuffle
from tools import SimpleTransformer

class LexiconSet:
    def __init__(self, lexicon_path):
	
        self.lexicon = np.array([line.rstrip("\n").split(",") for line in open(lexicon_path, "r").readlines()]).flatten()
	self.num_label = len(set(self.lexicon))  
    
    def getLexicon(self):
        return self.lexcion

    def getNumLabel(self):
        return self.num_label 

class MultilabelDataLayerSync(caffe.Layer):

    def setup(self, bottom, top):	
        
        params = eval(self.param_str)
        
	self.top_names = ['data', 'label']
        self.batch_size = params['batch_size']
        self.img_channel = params['channel']
        self.lexicon = LexiconSet(params['lexicon'])
        self.batch_loader = BatchLoader(params, self.lexicon, self.img_channel)

        top[0].reshape(self.batch_size, self.img_channel, params['im_shape'][0], params['im_shape'][1])
        top[1].reshape(self.batch_size, self.lexicon.getNumLabel())
        

    def forward(self, bottom, top):

        for itt in range(self.batch_size):
            im, multilabel = self.batch_loader.load_next_image()
	    top[0].data[itt, ...] = im
            top[1].data[itt, ...] = multilabel

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass


class BatchLoader(object):

    def __init__(self, params, lexicon, channel):

        self.im_shape = params['im_shape']
        self.data_folder = params['data_folder']
	self.lexicon = lexicon
	self.channel = channel

        self.indexlist = [line.rstrip('\n').split(",") for line in open(osp.join(self.data_folder, params['split'] + '.txt'), 'r').readlines()]

        self._cur = 0  
        self.transformer = SimpleTransformer(channel)

        print "BatchLoader initialized with {} images".format(len(self.indexlist))

    def load_next_image(self):

        if self._cur == len(self.indexlist):
            self._cur = 0
            shuffle(self.indexlist)
	

	img_annotations = [self.indexlist[self._cur][i] for i in range(1, len(self.indexlist[self._cur]))]        
        img_path = self.indexlist[self._cur][0]

        im = scipy.misc.imread(img_path, flatten=(self.channel==1))
        im = scipy.misc.imresize(im, self.im_shape)      

	try:
		multilabel = np.zeros(self.lexicon.getNumLabel())
		for a in img_annotations:
			multilabel[int(a)] = 1.0
	except:
		print "load error!!", self.indexlist[self._cur]

        self._cur += 1
        return self.transformer.preprocess(im), multilabel


def check_params(params):
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'pascal_root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['split'],
        params['batch_size'],
        params['im_shape'])
