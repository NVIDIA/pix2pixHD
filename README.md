# README-换脸模型
此README文件对应本次报告中基于pix2pix模型修改的换脸代码，包括在pix2pix-HD模型基础上主要的修改部分以及代码的使用方式。
在这里我们只给出效果最好的换脸代码（详见报告），报告中提到的效果较差的代码不在这里赘述。

## 主要修改部分
1. `models/pix2pixHD_model.py`中添加了中心对齐操作的代码（line 322-340）。
2. `models/pix2pixHD_model.py`中添加了IMNloss约束特征融合网络生成效果，并添加encoder生成feat的inclass_loss和betweenclass_loss（详见代码），使用到的cosin metric也在此文件中有定义。
3. `models/networks.py中`定义了IdentityNetwork（line 255），对应报告中的IMN。

## 使用方式
1. 使用命令行：
  ```
  CUDA_VISIBLE_DEVICES=(index of device) python train.py --name (your training name) --label_nc 19 --loadSize (your input size, default=512) --dataroot ./datasets/(your datafile)/ --label_feat
  ```
  以完成训练，同样需要提前获取label和inst。

2. 查看结果：
  在checkpoints中查看训练结果。

3. 进行测试：（这一部分代码我不了解，你最后修改一下参数，改成括号形式的）
     - 先生成目标图片feature，实验名需要同名
       ```
       python encode_features.py --name (your testing_feat name) --dataroot ./datasets/(your datafile)/ --label_nc 19 --loadSize (your input size) --label_feat --load_pretrain ./checkpoints/(your testing_feat name)/
       ```
     - pretrain模型要放在同名的/chechpoints/下面，会自动检索
       ```
       cp ./checkpoints/(your training name)/latest_net_*.pth ./checkpoints/(your testing_feat name)/
       ```
     - 进行测试
       ```
       CUDA_VISIBLE_DEVICES=(index of device) python test.py --name (your testing name) --label_nc 19 --loadSize (your input size) --dataroot ./datasets/(your datafile)/ --label_feat --random_pair 0 --load_pretrain ./checkpoints/(your training name)
       ```

## 语义分割模型使用方法
1. 源代码仓库下载：https://github.com/zllrunning/face-parsing.PyTorch
2. 预训练模型下载：https://jbox.sjtu.edu.cn/l/91LFbn ；放在 `./res/cp` 中
3. 路径更改：`test.py` 中 `dspth=[dir_of_testset]`
4. conda环境下运行 `python test.py`，10张img约用时4.44秒
5. 生成后的label图片在 `./res/test_res` 中