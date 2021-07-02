# ReviewKD using PaddlePaddle

This tutorial introduces how to train `MobileNetV1` using ReviewKD base on PaddlePaddle.

For more details of image classification models based on PaddlePaddle, please refer to [PaddleClas](https://github.com/PaddlePaddle/PaddleClas).

## 1. Prepare for the environment.

Please refer to [installation](./install_en.md) to install PaddlePaddle.

Then use the following command to install the dependencies.

```shell
pip3.7 install --upgrade -r requirements.txt
```

## 2. Prepare for the dataset and pretrained model

### 2.1 Prepare for the ImageNet dataset.

You can place the ImageNet dataset into folder `dataset`. The structure is as follows.

```
├── dataset/ILSVRC2012
│   ├── train/
│   ├── val/
│   ├── train_list.txt
│   ├── val_list.txt
├── ...
```

### 2.2 Prepare for the pretrained model.

Using the following command to download the pretrained model of `ResNet50`, which is the teacher model in this task.

```shell
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_pretrained.pdparams
```

## 3. Model training and evaluation.

### 3.1 Model training

You can use the following command to run the training process.

```shell
python tools/train.py -c configs/r50_mv1_reviewkd.yaml
```

If you want to train the model using multiple gpus, you can use the following command.

```shell
python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    --log_dir="./log_train" \
    tools/train.py \
        -c configs/r50_mv1_reviewkd.yaml
```

### 3.2 Model evaluation

You can use the following command to evaluate the model.

```shell
python3.7 tools/eval.py -c configs/r50_mv1_reviewkd.yaml -o pretrained_model="./output/DistillationModel/best_model/ppcls"
```

If you want to evaluate the model using multiple gpus, you can use the following command.

```shell
python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    --log_dir="./log_eval" \
    tools/eval.py \
        -c configs/r50_mv1_reviewkd.yaml \
        -o pretrained_model="./output/DistillationModel/best_model/ppcls"
```

## 4. Experiments

The following table shows results of the distillation method using PaddlePaddle.

| Method   | Student             | Teacher          | Top1 acc |
|----------|---------------------|------------------|----------|
| ReviewKD | MobileNetV1 (68.8%) | ResNet50 (76.5%) | 72.7%   |
