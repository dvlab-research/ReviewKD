#!/usr/bin/env bash

# if the pretrained model is not downloaded
# you can use the following command to download
# wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_pretrained.pdparams

python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    --log_dir="./log_train" \
    tools/train.py \
        -c configs/r50_mv1_reviewkd.yaml \
        -o model_save_dir="./output/"


