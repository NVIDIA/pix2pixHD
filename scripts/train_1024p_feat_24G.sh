############## To train images at 2048 x 1024 resolution after training 1024 x 512 resolution models #############
######## Using GPUs with 24G memory
# First precompute feature maps and save them
python precompute_feature_maps.py --name label2city_512p_feat;
# Adding instances and encoded features
python train.py --name label2city_1024p_feat --netG local --ngf 32 --num_D 3 --load_pretrain checkpoints/label2city_512p_feat/ --niter 50 --niter_decay 50 --niter_fix_global 10 --resize_or_crop none --instance_feat --load_features