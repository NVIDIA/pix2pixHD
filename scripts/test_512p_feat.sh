################################ Testing ################################
# first precompute and cluster all features
python encode_features.py --name label2city_512p_feat;
# use instance-wise features
python test.py --name label2city_512p_feat --instance_feat