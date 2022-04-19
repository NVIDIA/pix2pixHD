from contextlib import closing
from functools import total_ordering
import numpy as np
import torch
import os
import time
import sys
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
sys.path.append("..")
from faceparsing.logger import setup_logger
from faceparsing.model import BiSeNet
import torch.nn as nn
import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2


class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        use_importance=1
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True,use_importance)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake,importance_loss):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake,importance_loss),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc        
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num                  
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)        

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        # Feature merge network
        self.imn=networks.IdentityNetwork()
        ### Encoder network
        if self.gen_features:          
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)  
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            # pretrained_path = '' if not self.isTrain else opt.load_pretrain
            pretrained_path = opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.imn, 'I', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
                
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake','importance_loss')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())  
                params += list(self.imn.parameters()) 
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            
            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):          
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)         
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())
            if self.opt.label_feat:
                inst_map = label_map.cuda()

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)
    def get_importance(self,semantic):
        semantic_list = np.unique(semantic.cpu().numpy().astype(int))
        importance = np.zeros([1,1,512,512])
        min_pix_num=512*512
        for i in semantic_list:
            semantic_np=semantic.cpu().numpy()
            list_of_semantic_i=np.where(semantic_np[0][0]==i)
            pix_num=len(list_of_semantic_i[0])
            #print("pix_num",pix_num,"   list_of_semantic_length",len(list_of_semantic_i[0]))
            for pos in range(len(list_of_semantic_i[0])):
                importance[0][0][list_of_semantic_i[0][pos]][list_of_semantic_i[1][pos]]=pix_num
            min_pix_num=min(pix_num,min_pix_num)
        importance=torch.from_numpy(importance/min_pix_num)
        #print(importance)
        return importance
    def vis_parsing_maps(im, parsing_anno, stride):
    # Colors for all 20 parts
    
        im = np.array(im)
        vis_im = im.copy().astype(np.uint8)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        return vis_parsing_anno


    def get_semantic(self,fake_image, cp='79999_iter.pth'):
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = osp.join('faceparsing/res/cp', cp)
        net.load_state_dict(torch.load(save_pth))
        net.eval()

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        with torch.no_grad():
            fake_image=(fake_image+1)/2*255.0
            fake_image=torch.clamp(fake_image,0,255)
            toPIL = transforms.ToPILImage()#这里维度顺序不确定
            image=toPIL(fake_image.squeeze())
            image = image.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img=torch.unsqueeze(img,0)
            img = img.cuda()
            #print("image shape",img.shape) (1,3,512,512)
            print("nonzero in fake img ",torch.unique(fake_image))
            #print(img.shape)
            out = net(img)[0]
            #print("out_is",out)
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            print(np.unique(parsing))
            vis_parsing_anno = parsing.copy().astype(np.uint8)
            vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
            trans=transforms.ToTensor()
            semantic=trans(vis_parsing_anno).unsqueeze(0)
            
            return semantic#不是灰度图？

    def forward(self, label, inst, image, feat,source_label,source_inst,source_image,source_feat, infer=False):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)  
        source_input_label, source_inst_map, source_real_image, source_feat_map = self.encode_input(source_label, source_inst,
        source_image, source_feat)
        # Fake Generation
        if self.use_features:
            if not self.opt.load_features:
                #print("real_inst_size",inst_map.shape)
                #print("real_image_size",real_image.shape)
                feat_map = self.netE.forward(real_image, inst_map) 
                source_feat_map = self.netE.forward(source_real_image, source_inst_map) 
                input_for_merge=torch.cat((torch.cat((feat_map,source_feat_map),dim=3),torch.cat((source_feat_map,feat_map),dim=3)),dim=2)
                merged_feat_map= self.imn(input_for_merge)
            input_concat = torch.cat((input_label, merged_feat_map), dim=1)                        
        else:
            input_concat = input_label
        fake_image = self.netG.forward(input_concat)
        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):#此处只减少第一层的loss
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat

        use_importance=1
        importance_loss=0
        #fake_semantic=self.get_semantic(fake_image)
        #print(torch.unique(fake_semantic))
        if(use_importance):
            
            # Importance loss added by maoliyuan
            #fake_semantic=self.get_semantic(fake_image)
        #print("fake_semantic_size_is",fake_semantic.shape)
            fake_feat=self.netE.forward(fake_image,inst_map)
            importance_true=self.get_importance(inst_map).cuda()
            #importance_fake=self.get_importance(fake_semantic).cuda()
            #importance=importance_true*importance_fake
            inverse_importance=importance_true/torch.max(importance_true)
            importance=1.0/importance_true
            importance=importance.cuda()
            inverse_importance=inverse_importance.cuda()
            importance_loss_fn = torch.nn.MSELoss(reduction='sum')
            identityloss=nn.L1Loss()
            identitylabel=[1,2,3,4,5,7,8,10,11,12,13]
            pos={}
            source_inst_squeeze=torch.squeeze(source_inst_map)
            inst_squeeze=torch.squeeze(inst_map)
            for label in identitylabel:
                flag=(source_inst_squeeze==label)
                flag_list=flag.nonzero()
                if(len(flag_list)!=0):
                    pos[label]=flag_list[0]
            # inverse_loss_fn=torch.nn.MSELoss(reduction='sum')
            #print(torch.sum(torch.isnan(feat_map).int()).item())
            #print("feat_map",feat_map.shape)
            # feature_diff=0
            inclass_loss=0
            betweenclass_loss=0
            id_betweenclass_loss=0
            imn_loss=0
            fake_important_id_area=torch.zeros([512,512]).cuda()
            source_important_id_area=torch.zeros([512,512]).cuda()
            fake_important_id_area=fake_important_id_area+torch.where((inst_map[0][0]<=13) * (inst_map[0][0]!=6) * (inst_map[0][0]!=9),1,0)
            source_important_id_area=source_important_id_area+torch.where((source_inst_map[0][0]<=13) * (source_inst_map[0][0]!=6) * (source_inst_map[0][0]!=9),1,0)
            center_area_source=source_inst_squeeze==10
            center_area_source_pos=torch.mean(center_area_source.nonzero().float(),dim=0)
            center_area_target=inst_squeeze==10
            center_area_target_pos=torch.mean(center_area_target.nonzero().float(),dim=0)
            move_dist=center_area_target_pos-center_area_source_pos
            #print("center_area_pos",move_dist)
            for i in range(3):
                idealidentity=torch.zeros([512,512]).cuda()
                if(move_dist[0]<=0):
                    source_inst_narrow=source_inst_map[0][0].narrow(0,int(-move_dist[0]),512-int(-move_dist[0]))
                    concate=torch.zeros([int(-move_dist[0]),512]).cuda()
                    source_inst_moved_dim0=torch.cat([source_inst_narrow,concate],dim=0)
                    #print("cocate_size",source_inst_moved_dim0.shape)
                else:
                    source_inst_narrow=source_inst_map[0][0].narrow(0,0,512-int(move_dist[0]))
                    concate=torch.zeros([int(move_dist[0]),512]).cuda()
                    source_inst_moved_dim0=torch.cat([concate,source_inst_narrow],dim=0)
                if(move_dist[1]<=0):
                    source_inst_narrow_dim1=source_inst_moved_dim0.narrow(1,int(-move_dist[1]),512-int(-move_dist[1]))
                    concate=torch.zeros([512,int(-move_dist[1])]).cuda()
                    source_inst_moved=torch.cat([source_inst_narrow_dim1,concate],dim=1)
                else:
                    source_inst_narrow_dim1=source_inst_moved_dim0.narrow(1,0,512-int(move_dist[1]))
                    concate=torch.zeros([512,int(move_dist[1])]).cuda()
                    source_inst_moved=torch.cat([concate,source_inst_narrow_dim1],dim=1)
                idealidentity=torch.where((fake_important_id_area!=0) * (source_important_id_area==0),feat_map[0][i].double(),float(0))
                idealidentity=idealidentity+fake_important_id_area*source_inst_moved
                idealidentity=idealidentity.detach()
                idealidentity = idealidentity.to(torch.float32)
                idealidentity_with_background=torch.where(idealidentity==0,feat_map[0][i],idealidentity)
                # source_important_id_area=torch.zeros([512,512]).cuda()
                # source_important_id_area=source_important_id_area+torch.where((source_inst_map[0][0]<=13) * (source_inst_map[0][0]!=6) * (source_inst_map[0][0]!=9),1,0)
                # for label in identitylabel:
                #     if(label in pos):
                #         replace=source_feat_map[0][i][pos[label][0]][pos[label][1]]
                #         idealidentity=idealidentity+torch.where(inst_map==label,float(replace),float(0))
                #idealidentity=torch.where(idealidentity==0,feat_map[0][i],idealidentity)
                # feature_diff=feature_diff+1-(torch.sum(source_feat_map[0][i] * feat_map[0][i]) / (torch.norm(source_feat_map[0][i]) * torch.norm(feat_map[0][i])))
                #balance the loss
                # total_importance=torch.sum(importance)
                # total_inverse_importance=torch.sum(inverse_importance)
                # balance_ratio=total_inverse_importance/(total_importance+total_inverse_importance)
                #print("feat_max",torch.max(source_feat_map))
                imn_loss=imn_loss+1-torch.sum(merged_feat_map[0][i] *idealidentity_with_background) / (torch.norm(idealidentity_with_background) * torch.norm(merged_feat_map[0][i]))
                inclass_loss=inclass_loss+1-torch.sum(idealidentity *fake_feat[0][i]) / (torch.norm(fake_important_id_area*fake_feat[0][i]) * torch.norm(idealidentity))
                source_id=source_feat_map[0][i]*source_important_id_area
                target_id=feat_map[0][i]*fake_important_id_area
                id_betweenclass_loss=id_betweenclass_loss+\
                    torch.sum(source_id * target_id) /(torch.norm(source_id) * torch.norm(target_id))
                betweenclass_loss=betweenclass_loss+torch.sum(source_feat_map[0][i] * feat_map[0][i]) / (torch.norm(source_feat_map[0][i]) * torch.norm(feat_map[0][i]))
                importance_loss=importance_loss+inclass_loss+0.5*betweenclass_loss+id_betweenclass_loss+0.5*imn_loss
                
                            #importance_loss_fn(inverse_importance*feat_map[0][i],inverse_importance*fake_feat[0][i])
                            
            #print("between_clase_loss_is",betweenclass_loss.item())
            #print("in_clase_loss_is",inclass_loss.item())
            # print("feature_diff",feature_diff)
            
            
        


        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake ,importance_loss), None if not infer else fake_image ]

    def inference(self, label, inst, image, s_label, s_inst, s_image):
        # Encode Inputs
        label = Variable(label)
        inst = Variable(inst)
        image = Variable(image)
        s_label = Variable(s_label)
        s_inst = Variable(s_inst)
        s_image = Variable(s_image)
        input_label, inst_map, real_image, _ = self.encode_input(label, inst, image, infer=True)
        s_input_label, s_inst_map, s_real_image, _ = self.encode_input(s_label, s_inst, s_image, infer=True)

        # Fake Generation
        if self.use_features:  #用label_feat等
            # encode the real image to get feature map
            s_feat_map = self.netE.forward(s_real_image, s_inst_map)
            feat_map = self.netE.forward(real_image, inst_map)  #除此之外没用image
            input_for_merge=torch.cat((torch.cat((feat_map,s_feat_map),dim=3),torch.cat((s_feat_map,feat_map),dim=3)),dim=2)
            self.imn.cuda()
            merged_feat_map= self.imn(input_for_merge)
            input_concat = torch.cat((input_label, merged_feat_map), dim=1)
        else:
            input_concat = input_label
           
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path, encoding='latin1', allow_pickle=True).item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].item()
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        self.save_network(self.imn, 'I', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)

        
