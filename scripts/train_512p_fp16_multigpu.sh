######## Multi-GPU training example #######
python -m torch.distributed.launch train.py --name label2city_512p --batchSize 8 --gpu_ids 0,1,2,3,4,5,6,7 --fp16