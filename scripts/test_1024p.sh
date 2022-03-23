#!/bin/bash
################################ Testing ################################
# labels only
pix2pixhd-test --name label2city_1024p --netG local --ngf 32 --resize_or_crop none "$@"
