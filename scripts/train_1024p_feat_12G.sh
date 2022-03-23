#!/bin/sh
############## To train images at 2048 x 1024 resolution after training 1024 x 512 resolution models #############
##### Using GPUs with 12G memory (not tested)
# First precompute feature maps and save them
pix2pixhd-precompute-feature-maps --name label2city_512p_feat;
# Adding instances and encoded features
pix2pixhd-train --name label2city_1024p_feat --netG local --ngf 32 --num_D 3 --load_pretrain checkpoints/label2city_512p_feat/ --niter_fix_global 20 --resize_or_crop crop --fineSize 896 --instance_feat --load_features
