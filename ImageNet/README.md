## Review KD models on ImageNet

### Environment

We verify our code on 
* 4x2080Ti GPUs
* CUDA 10.1
* python 3.7
* torch 1.6.0
* torchvision 0.7.0

Other similar envirouments should also work properly.

### Installation

We use apex, please install apex refer to https://github.com/NVIDIA/apex 
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Results

| Student    | Teacher  | Baseline | Ours  |
|------------|----------|----------|-------|
| ResNet18   | ResNet34 | 69.75    | 71.61 |
| MobileNet  | ResNet50 | 68.87    | 72.56 |

### Training

Use the following command to train resnet-18 with ReviewKD
```
python -m torch.distributed.launch --nproc_per_node=4 imagenet_amp.py \
    -a resnet18 --save_dir output/r18-r34/ \
    -b 64 -j 24 -p 100 \
    --teacher resnet34 \
    --review-kd-loss-weight 1.0 \
    path-to-ImageNet
```

Use the following command to train mobilenet with ReviewKD
```
python -m torch.distributed.launch --nproc_per_node=4 imagenet_amp.py \
    -a mobilenet --save_dir output/mv2-r50/ \
    -b 64 -j 24 -p 100 \
    --teacher resnet50 \
    --review-kd-loss-weight 8.0 \
    path-to-ImageNet    
```
