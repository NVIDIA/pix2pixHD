#!/bin/sh
### Using labels only
python -m torch.distributed.launch pix2pixhd/train.py --name label2city_512p --fp16
