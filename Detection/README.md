## Review KD models on COCO detection and instance segmentation

### Environment

We verify our code on 
* 4x2080Ti GPUs
* CUDA 10.1
* python 3.7
* torch 1.6.0
* torchvision 0.7.0

Other similar envirouments should also work properly.

### Installation

Our code is based on Detectron2, please install Detectron2 refer to https://github.com/facebookresearch/detectron2.

Please put the [COCO](https://cocodataset.org/#download) dataset in datasets/.

Please put the pretrained weights for teacher and student in pretrained/. You can find the pretrained weights [here](https://github.com/dvlab-research/ReviewKD/releases/). The pretrained models we provided contains both teacher's and student's weights. The teacher's weights come from Detectron2's pretrained detector. The student's weights are ImageNet pretrained weights.

### Training

Use the following command to train Faster-RCNN-FPN-R50 with ReviewKD for COCO Detection
```
python train_net.py --config-file configs/ReviewKD-R50-R101.yaml --num-gpus 4
```

Use the following command to train Faster-RCNN-FPN-R50 with ReviewKD for COCO Instance Segmentation
```
python train_net.py --config-file configs/ReviewKD-R50-R101-Mask.yaml --num-gpus 4
```

Use the following command to train Faster-RCNN-FPN-MobileNetV2 with ReviewKD for COCO Detection
```
python train_net.py --config-file configs/ReviewKD-MV2-R50.yaml --num-gpus 4
```

Use the following command to train Faster-RCNN-FPN-MobileNetV2 with ReviewKD for COCO Instance Segmentation
```
python train_net.py --config-file configs/ReviewKD-MV2-R50-Mask.yaml --num-gpus 4
```
