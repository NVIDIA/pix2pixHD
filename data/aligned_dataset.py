import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import random

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

         # source and target use same dataset, same path
        ### input A (label maps of source)
        dir_A = '_source_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images of source)
        dir_B = '_source_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps of source
        if not opt.no_instance:
            self.dir_C = os.path.join(opt.dataroot, opt.phase + '_source_inst')
            self.C_paths = sorted(make_dataset(self.dir_C))

        if not self.opt.random_pair:  # the pair is fixed, use in test mode usually
            ### input C (label maps of target)
            dir_D = '_target_label'
            self.dir_D = os.path.join(opt.dataroot, opt.phase + dir_D)
            self.D_paths = sorted(make_dataset(self.dir_D))

            ### input D (real images of target)
            dir_E = '_target_img'
            self.dir_E = os.path.join(opt.dataroot, opt.phase + dir_E)
            self.E_paths = sorted(make_dataset(self.dir_E))

            ### instance maps of source
            if not opt.no_instance:
                self.dir_F = os.path.join(opt.dataroot, opt.phase + '_target_inst')
                self.F_paths = sorted(make_dataset(self.dir_F))
            
            assert len(self.A_paths) == len(self.D_paths)

        self.dataset_size = len(self.A_paths) 
        self.mk = np.zeros([self.dataset_size], bool)


    def __get_set1__(self, index, s=True):
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            # 把所有label做了个resize后的列表。因为opt调了512所以就是原图*255
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = C_tensor = feat_tensor = 0
        ### input B (real images)
        B_path = self.B_paths[index]
        B = Image.open(B_path).convert('RGB')
        transform_B = get_transform(self.opt, params)
        B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            C_path = self.C_paths[index]
            C = Image.open(C_path)
            C_tensor = transform_A(C)  #inst不*255

        if s:
            source_dict = {'s_label': A_tensor, 's_inst': C_tensor, 's_image': B_tensor, 
                          's_feat': feat_tensor, 's_path': A_path}
            return source_dict
        else:
            target_dict = {'t_label': A_tensor, 't_inst': C_tensor, 't_image': B_tensor, 
                          't_feat': feat_tensor, 't_path': A_path}
            return target_dict

    def __get_set2__(self, index):
        ### input A (label maps)
        D_path = self.D_paths[index]              
        D = Image.open(D_path)        
        params = get_params(self.opt, D.size)
        if self.opt.label_nc == 0:
            transform_D = get_transform(self.opt, params)
            D_tensor = transform_D(D.convert('RGB'))
        else:
            transform_D = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            D_tensor = transform_D(D) * 255.0

        E_tensor = F_tensor = feat_tensor = 0
        ### input B (real images)
        E_path = self.E_paths[index]
        E = Image.open(E_path).convert('RGB')
        transform_E = get_transform(self.opt, params)
        E_tensor = transform_E(E)

        ### if using instance maps        
        if not self.opt.no_instance:
            F_path = self.F_paths[index]
            F = Image.open(F_path)
            F_tensor = transform_D(F)  #inst不*255

        target_dict = {'t_label': D_tensor, 't_inst': F_tensor, 't_image': E_tensor,
                      't_feat': feat_tensor, 't_path': D_path}
        return target_dict



    def __getitem__(self, index):
        input_dict = self.__get_set1__(index)
        if self.opt.random_pair:
            zidx = np.array(np.where(self.mk == False))
            zidx = zidx.squeeze()
            if len(zidx) == 0:
                self.mk.fill(0)
                zidx = np.arange(self.dataset_size, dtype='int64')
            rand = int(random.random() * len(zidx))
            index2 = zidx[rand]
            self.mk[index2] = True
            target_dict = self.__get_set1__(index2, s=False)
        else:
            target_dict = self.__get_set2__(index)
        
        input_dict.update(target_dict)
        return input_dict
        # input_dict = {'s_label', 's_inst', 's_image', 's_feat', 's_path', 
        #               't_label', 't_inst', 't_image', 't_feat', 't_path'}

    def __len__(self):
        return len(self.A_paths) #// self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'