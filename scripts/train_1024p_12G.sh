############## To train images at 2048 x 1024 resolution after training 1024 x 512 resolution models #############
##### Using GPUs with 12G memory (not tested)
# Using labels only
python train.py --name label2city_1024p --netG local --ngf 32 --num_D 3 --load_pretrain checkpoints/label2city_512p/ --niter_fix_global 20 --resize_or_crop crop --fineSize 1024