import torch
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import os
from PIL import Image
import util.util as util
from .base_model import BaseModel
from . import networks

class UIModel(BaseModel):
    def name(self):
        return 'UIModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.use_features = opt.instance_feat or opt.label_feat

        netG_input_nc = opt.label_nc
        if not opt.no_instance:
            netG_input_nc += 1            
        if self.use_features:   
            netG_input_nc += opt.feat_num           

        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)            
        self.load_network(self.netG, 'G', opt.which_epoch)

        print('---------- Networks initialized -------------')

    def toTensor(self, img, normalize=False):
        tensor = torch.from_numpy(np.array(img, np.int32, copy=False))
        tensor = tensor.view(1, img.size[1], img.size[0], len(img.mode))    
        tensor = tensor.transpose(1, 2).transpose(1, 3).contiguous()
        if normalize:
            return (tensor.float()/255.0 - 0.5) / 0.5        
        return tensor.float()

    def load_image(self, label_path, inst_path, feat_path):
        opt = self.opt
        # read label map
        label_img = Image.open(label_path)    
        if label_path.find('face') != -1:
            label_img = label_img.convert('L')
        ow, oh = label_img.size    
        w = opt.loadSize
        h = int(w * oh / ow)    
        label_img = label_img.resize((w, h), Image.NEAREST)
        label_map = self.toTensor(label_img)           
        
        # onehot vector input for label map
        self.label_map = label_map.cuda()
        oneHot_size = (1, opt.label_nc, h, w)
        input_label = self.Tensor(torch.Size(oneHot_size)).zero_()
        self.input_label = input_label.scatter_(1, label_map.long().cuda(), 1.0)

        # read instance map
        if not opt.no_instance:
            inst_img = Image.open(inst_path)        
            inst_img = inst_img.resize((w, h), Image.NEAREST)            
            self.inst_map = self.toTensor(inst_img).cuda()
            self.edge_map = self.get_edges(self.inst_map)          
            self.net_input = Variable(torch.cat((self.input_label, self.edge_map), dim=1), volatile=True)
        else:
            self.net_input = Variable(self.input_label, volatile=True)  
        
        self.features_clustered = np.load(feat_path).item()
        self.object_map = self.inst_map if opt.instance_feat else self.label_map 
                       
        object_np = self.object_map.cpu().numpy().astype(int) 
        self.feat_map = self.Tensor(1, opt.feat_num, h, w).zero_()                 
        self.cluster_indices = np.zeros(self.opt.label_nc, np.uint8)
        for i in np.unique(object_np):    
            label = i if i < 1000 else i//1000
            if label in self.features_clustered:
                feat = self.features_clustered[label]
                np.random.seed(i+1)
                cluster_idx = np.random.randint(0, feat.shape[0])
                self.cluster_indices[label] = cluster_idx
                idx = (self.object_map == i).nonzero()                    
                self.set_features(idx, feat, cluster_idx)

        self.net_input_original = self.net_input.clone()        
        self.label_map_original = self.label_map.clone()
        self.feat_map_original = self.feat_map.clone()
        if not opt.no_instance:
            self.inst_map_original = self.inst_map.clone()        

    def reset(self):
        self.net_input = self.net_input_prev = self.net_input_original.clone()        
        self.label_map = self.label_map_prev = self.label_map_original.clone()
        self.feat_map = self.feat_map_prev = self.feat_map_original.clone()
        if not self.opt.no_instance:
            self.inst_map = self.inst_map_prev = self.inst_map_original.clone()
        self.object_map = self.inst_map if self.opt.instance_feat else self.label_map 

    def undo(self):        
        self.net_input = self.net_input_prev
        self.label_map = self.label_map_prev
        self.feat_map = self.feat_map_prev
        if not self.opt.no_instance:
            self.inst_map = self.inst_map_prev
        self.object_map = self.inst_map if self.opt.instance_feat else self.label_map 
            
    # get boundary map from instance map
    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        return edge.float()

    # change the label at the source position to the label at the target position
    def change_labels(self, click_src, click_tgt): 
        y_src, x_src = click_src[0], click_src[1]
        y_tgt, x_tgt = click_tgt[0], click_tgt[1]
        label_src = int(self.label_map[0, 0, y_src, x_src])
        inst_src = self.inst_map[0, 0, y_src, x_src]
        label_tgt = int(self.label_map[0, 0, y_tgt, x_tgt])
        inst_tgt = self.inst_map[0, 0, y_tgt, x_tgt]

        idx_src = (self.inst_map == inst_src).nonzero()         
        # need to change 3 things: label map, instance map, and feature map
        if idx_src.shape:
            # backup current maps
            self.backup_current_state() 

            # change both the label map and the network input
            self.label_map[idx_src[:,0], idx_src[:,1], idx_src[:,2], idx_src[:,3]] = label_tgt
            self.net_input[idx_src[:,0], idx_src[:,1] + label_src, idx_src[:,2], idx_src[:,3]] = 0
            self.net_input[idx_src[:,0], idx_src[:,1] + label_tgt, idx_src[:,2], idx_src[:,3]] = 1                                    
            
            # update the instance map (and the network input)
            if inst_tgt > 1000:
                # if different instances have different ids, give the new object a new id
                tgt_indices = (self.inst_map > label_tgt * 1000) & (self.inst_map < (label_tgt+1) * 1000)
                inst_tgt = self.inst_map[tgt_indices].max() + 1
            self.inst_map[idx_src[:,0], idx_src[:,1], idx_src[:,2], idx_src[:,3]] = inst_tgt
            self.net_input[:,-1,:,:] = self.get_edges(self.inst_map)

            # also copy the source features to the target position      
            idx_tgt = (self.inst_map == inst_tgt).nonzero()    
            if idx_tgt.shape:
                self.copy_features(idx_src, idx_tgt[0,:])

        self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))

    # add strokes of target label in the image
    def add_strokes(self, click_src, label_tgt, bw, save):
        # get the region of the new strokes (bw is the brush width)        
        size = self.net_input.size()
        h, w = size[2], size[3]
        idx_src = torch.LongTensor(bw**2, 4).fill_(0)
        for i in range(bw):
            idx_src[i*bw:(i+1)*bw, 2] = min(h-1, max(0, click_src[0]-bw//2 + i))
            for j in range(bw):
                idx_src[i*bw+j, 3] = min(w-1, max(0, click_src[1]-bw//2 + j))
        idx_src = idx_src.cuda()
        
        # again, need to update 3 things
        if idx_src.shape:
            # backup current maps
            if save:
                self.backup_current_state()

            # update the label map (and the network input) in the stroke region            
            self.label_map[idx_src[:,0], idx_src[:,1], idx_src[:,2], idx_src[:,3]] = label_tgt
            for k in range(self.opt.label_nc):
                self.net_input[idx_src[:,0], idx_src[:,1] + k, idx_src[:,2], idx_src[:,3]] = 0
            self.net_input[idx_src[:,0], idx_src[:,1] + label_tgt, idx_src[:,2], idx_src[:,3]] = 1                 

            # update the instance map (and the network input)
            self.inst_map[idx_src[:,0], idx_src[:,1], idx_src[:,2], idx_src[:,3]] = label_tgt
            self.net_input[:,-1,:,:] = self.get_edges(self.inst_map)
            
            # also update the features if available
            if self.opt.instance_feat:                                            
                feat = self.features_clustered[label_tgt]
                #np.random.seed(label_tgt+1)   
                #cluster_idx = np.random.randint(0, feat.shape[0])
                cluster_idx = self.cluster_indices[label_tgt]
                self.set_features(idx_src, feat, cluster_idx)                                                  
        
        self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))

    # add an object to the clicked position with selected style
    def add_objects(self, click_src, label_tgt, mask, style_id=0):
        y, x = click_src[0], click_src[1]
        mask = np.transpose(mask, (2, 0, 1))[np.newaxis,...]        
        idx_src = torch.from_numpy(mask).cuda().nonzero()        
        idx_src[:,2] += y
        idx_src[:,3] += x

        # backup current maps
        self.backup_current_state()

        # update label map
        self.label_map[idx_src[:,0], idx_src[:,1], idx_src[:,2], idx_src[:,3]] = label_tgt        
        for k in range(self.opt.label_nc):
            self.net_input[idx_src[:,0], idx_src[:,1] + k, idx_src[:,2], idx_src[:,3]] = 0
        self.net_input[idx_src[:,0], idx_src[:,1] + label_tgt, idx_src[:,2], idx_src[:,3]] = 1            

        # update instance map
        self.inst_map[idx_src[:,0], idx_src[:,1], idx_src[:,2], idx_src[:,3]] = label_tgt
        self.net_input[:,-1,:,:] = self.get_edges(self.inst_map)
                
        # update feature map
        self.set_features(idx_src, self.feat, style_id)                
        
        self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))

    def single_forward(self, net_input, feat_map):
        net_input = torch.cat((net_input, feat_map), dim=1)
        fake_image = self.netG.forward(net_input)

        if fake_image.size()[0] == 1:
            return fake_image.data[0]        
        return fake_image.data


    # generate all outputs for different styles
    def style_forward(self, click_pt, style_id=-1):           
        if click_pt is None:            
            self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))
            self.crop = None
            self.mask = None        
        else:                       
            instToChange = int(self.object_map[0, 0, click_pt[0], click_pt[1]])
            self.instToChange = instToChange
            label = instToChange if instToChange < 1000 else instToChange//1000        
            self.feat = self.features_clustered[label]
            self.fake_image = []
            self.mask = self.object_map == instToChange
            idx = self.mask.nonzero()
            self.get_crop_region(idx)            
            if idx.size():                
                if style_id == -1:
                    (min_y, min_x, max_y, max_x) = self.crop
                    ### original
                    for cluster_idx in range(self.opt.multiple_output):
                        self.set_features(idx, self.feat, cluster_idx)
                        fake_image = self.single_forward(self.net_input, self.feat_map)
                        fake_image = util.tensor2im(fake_image[:,min_y:max_y,min_x:max_x])
                        self.fake_image.append(fake_image)    
                    """### To speed up previewing different style results, either crop or downsample the label maps
                    if instToChange > 1000:
                        (min_y, min_x, max_y, max_x) = self.crop                                                
                        ### crop                                                
                        _, _, h, w = self.net_input.size()
                        offset = 512
                        y_start, x_start = max(0, min_y-offset), max(0, min_x-offset)
                        y_end, x_end = min(h, (max_y + offset)), min(w, (max_x + offset))
                        y_region = slice(y_start, y_start+(y_end-y_start)//16*16)
                        x_region = slice(x_start, x_start+(x_end-x_start)//16*16)
                        net_input = self.net_input[:,:,y_region,x_region]                    
                        for cluster_idx in range(self.opt.multiple_output):  
                            self.set_features(idx, self.feat, cluster_idx)
                            fake_image = self.single_forward(net_input, self.feat_map[:,:,y_region,x_region])                            
                            fake_image = util.tensor2im(fake_image[:,min_y-y_start:max_y-y_start,min_x-x_start:max_x-x_start])
                            self.fake_image.append(fake_image)
                    else:
                        ### downsample
                        (min_y, min_x, max_y, max_x) = [crop//2 for crop in self.crop]                    
                        net_input = self.net_input[:,:,::2,::2]                    
                        size = net_input.size()
                        net_input_batch = net_input.expand(self.opt.multiple_output, size[1], size[2], size[3])             
                        for cluster_idx in range(self.opt.multiple_output):  
                            self.set_features(idx, self.feat, cluster_idx)
                            feat_map = self.feat_map[:,:,::2,::2]
                            if cluster_idx == 0:
                                feat_map_batch = feat_map
                            else:
                                feat_map_batch = torch.cat((feat_map_batch, feat_map), dim=0)
                        fake_image_batch = self.single_forward(net_input_batch, feat_map_batch)
                        for i in range(self.opt.multiple_output):
                            self.fake_image.append(util.tensor2im(fake_image_batch[i,:,min_y:max_y,min_x:max_x]))"""
                                        
                else:
                    self.set_features(idx, self.feat, style_id)
                    self.cluster_indices[label] = style_id
                    self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))        

    def backup_current_state(self):
        self.net_input_prev = self.net_input.clone()
        self.label_map_prev = self.label_map.clone() 
        self.inst_map_prev = self.inst_map.clone() 
        self.feat_map_prev = self.feat_map.clone() 

    # crop the ROI and get the mask of the object
    def get_crop_region(self, idx):
        size = self.net_input.size()
        h, w = size[2], size[3]
        min_y, min_x = idx[:,2].min(), idx[:,3].min()
        max_y, max_x = idx[:,2].max(), idx[:,3].max()             
        crop_min = 128
        if max_y - min_y < crop_min:
            min_y = max(0, (max_y + min_y) // 2 - crop_min // 2)
            max_y = min(h-1, min_y + crop_min)
        if max_x - min_x < crop_min:
            min_x = max(0, (max_x + min_x) // 2 - crop_min // 2)
            max_x = min(w-1, min_x + crop_min)
        self.crop = (min_y, min_x, max_y, max_x)           
        self.mask = self.mask[:,:, min_y:max_y, min_x:max_x]

    # update the feature map once a new object is added or the label is changed
    def update_features(self, cluster_idx, mask=None, click_pt=None):        
        self.feat_map_prev = self.feat_map.clone()
        # adding a new object
        if mask is not None:
            y, x = click_pt[0], click_pt[1]
            mask = np.transpose(mask, (2,0,1))[np.newaxis,...]        
            idx = torch.from_numpy(mask).cuda().nonzero()        
            idx[:,2] += y
            idx[:,3] += x    
        # changing the label of an existing object 
        else:            
            idx = (self.object_map == self.instToChange).nonzero()              

        # update feature map
        self.set_features(idx, self.feat, cluster_idx)        

    # set the class features to the target feature
    def set_features(self, idx, feat, cluster_idx):        
        for k in range(self.opt.feat_num):
            self.feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k] 

    # copy the features at the target position to the source position
    def copy_features(self, idx_src, idx_tgt):        
        for k in range(self.opt.feat_num):
            val = self.feat_map[idx_tgt[0], idx_tgt[1] + k, idx_tgt[2], idx_tgt[3]]
            self.feat_map[idx_src[:,0], idx_src[:,1] + k, idx_src[:,2], idx_src[:,3]] = val 

    def get_current_visuals(self, getLabel=False):                              
        mask = self.mask     
        if self.mask is not None:
            mask = np.transpose(self.mask[0].cpu().float().numpy(), (1,2,0)).astype(np.uint8)        

        dict_list = [('fake_image', self.fake_image), ('mask', mask)]

        if getLabel: # only output label map if needed to save bandwidth
            label = util.tensor2label(self.net_input.data[0], self.opt.label_nc)                    
            dict_list += [('label', label)]

        return OrderedDict(dict_list)