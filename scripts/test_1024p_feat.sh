################################ Testing ################################
# first precompute and cluster all features
pix2pix-encode-features --name label2city_1024p_feat --netG local --ngf 32 --resize_or_crop none;
# use instance-wise features
pix2pixhd-test --name label2city_1024p_feat ---netG local --ngf 32 --resize_or_crop none --instance_feat
