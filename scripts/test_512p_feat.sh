#!/bin/sh
################################ Testing ################################
# first precompute and cluster all features
pix2pixhd-encode-features --name label2city_512p_feat;
# use instance-wise features
pix2pixhd-test --name label2city_512p_feat --instance_feat
