################################ Testing ################################
# first precompute and cluster all features
python encode_features.py --name label2city_1024p_feat --netG local --ngf 32 --resize_or_crop none;
# use instance-wise features
python test.py --name label2city_1024p_feat ---netG local --ngf 32 --resize_or_crop none --instance_feat