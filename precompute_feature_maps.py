### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import os
import util.util as util
from torch.autograd import Variable
import torch.nn as nn

opt = TrainOptions().parse()
opt.nThreads = 1
opt.batchSize = 1 
opt.serial_batches = True 
opt.no_flip = True
opt.instance_feat = True

name = 'features'
save_path = os.path.join(opt.checkpoints_dir, opt.name)

############ Initialize #########
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
model = create_model(opt)
util.mkdirs(os.path.join(opt.dataroot, opt.phase + '_feat'))

######## Save precomputed feature maps for 1024p training #######
for i, data in enumerate(dataset):
	print('%d / %d images' % (i+1, dataset_size)) 
	feat_map = model.module.netE.forward(Variable(data['image'].cuda(), volatile=True), data['inst'].cuda())
	feat_map = nn.Upsample(scale_factor=2, mode='nearest')(feat_map)
	image_numpy = util.tensor2im(feat_map.data[0])
	save_path = data['path'][0].replace('/train_label/', '/train_feat/')
	util.save_image(image_numpy, save_path)