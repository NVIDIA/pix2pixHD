### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### label maps         
        self.dir_label = os.path.join(opt.dataroot, opt.phase + '_label')              
        self.label_paths = sorted(make_dataset(self.dir_label))

        ### real images
        if opt.isTrain:
            self.dir_image = os.path.join(opt.dataroot, opt.phase + '_img')  
            self.image_paths = sorted(make_dataset(self.dir_image))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.label_paths) 
      
    def __getitem__(self, index):        
        ### label maps        
        label_path = self.label_paths[index]              
        label = Image.open(label_path)        
        params = get_params(self.opt, label.size)
        if self.opt.label_nc == 0:
            transform_label = get_transform(self.opt, params)
            label_tensor = transform_label(label.convert('RGB'))
        else:
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            label_tensor = transform_label(label) * 255.0

        image_tensor = inst_tensor = feat_tensor = 0
        ### real images
        if self.opt.isTrain:
            image_path = self.image_paths[index]   
            image = Image.open(image_path).convert('RGB')
            transform_image = get_transform(self.opt, params)      
            image_tensor = transform_image(image)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_label(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_label(feat))                            

        input_dict = {'label': label_tensor, 'inst': inst_tensor, 'image': image_tensor, 
                      'feat': feat_tensor, 'path': label_path}

        return input_dict

    def __len__(self):
        return len(self.label_paths)

    def name(self):
        return 'AlignedDataset'