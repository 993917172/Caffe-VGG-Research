import numpy as np


class SimpleTransformer:

    def __init__(self, channel, mean=None, scale=1.0):
	if mean is None:
		mean = [128] * channel
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = scale
	self.channel = channel

    def set_mean(self, mean):
        self.mean = mean

    def set_scale(self, scale):
        self.scale = scale

    def preprocess(self, im):

        im = np.float32(im)
	if self.channel == 3:
	        im = im[:, :, ::-1]  # change to BGR

        im -= self.mean
        im *= self.scale
	
	if self.channel == 3:
	        im = im.transpose((2, 0, 1))

        return im

    def deprocess(self, im):

	if self.channel == 3:
	        im = im.transpose(1, 2, 0)

        im /= self.scale
        im += self.mean

	if self.channel == 3:        
		im = im[:, :, ::-1]  # change to RGB

        return np.uint8(im)


class CaffeSolver:

    def __init__(self, testnet_prototxt_path="testnet.prototxt",
                 trainnet_prototxt_path="trainnet.prototxt", debug=False):

        self.sp = {}

        # critical:
        self.sp['base_lr'] = '0.001'
        self.sp['momentum'] = '0.9'

        # speed:
        self.sp['test_iter'] = '100'
        self.sp['test_interval'] = '250'

        # looks:
        self.sp['display'] = '25'
        self.sp['snapshot'] = '2500'
        self.sp['snapshot_prefix'] = '"snapshot"'  # string within a string!

        # learning rate policy
        self.sp['lr_policy'] = '"fixed"'

        # important, but rare:
        self.sp['gamma'] = '0.1'
        self.sp['weight_decay'] = '0.0005'
        self.sp['train_net'] = '"' + trainnet_prototxt_path + '"'
        self.sp['test_net'] = '"' + testnet_prototxt_path + '"'

        # pretty much never change these.
        self.sp['max_iter'] = '100000'
        self.sp['test_initialization'] = 'false'
        self.sp['average_loss'] = '25'  # this has to do with the display.
        self.sp['iter_size'] = '1'  # this is for accumulating gradients

        if (debug):
            self.sp['max_iter'] = '12'
            self.sp['test_iter'] = '1'
            self.sp['test_interval'] = '4'
            self.sp['display'] = '1'

    def add_from_file(self, filepath):
        """
        Reads a caffe solver prototxt file and updates the Caffesolver
        instance parameters.
        """
        with open(filepath, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                splitLine = line.split(':')
                self.sp[splitLine[0].strip()] = splitLine[1].strip()

    def write(self, filepath):
        """
        Export solver parameters to INPUT "filepath". Sorted alphabetically.
        """
        f = open(filepath, 'w')
        for key, value in sorted(self.sp.items()):
            if not(type(value) is str):
                raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))
