#!/bin/bash
################################ Testing ################################
# labels only
python test.py --name label2city_1024p --netG local --ngf 32 --resize_or_crop none $@
