python train.py --model wrn-16-2 --teacher wrn-40-2 --teacher-weight checkpoints/cifar100_wrn-40-2__baseline1_best.pt --kd-loss-weight 5.0 --suffix reviewkd1

python train.py --model wrn-40-1 --teacher wrn-40-2 --teacher-weight checkpoints/cifar100_wrn-40-2__baseline1_best.pt --kd-loss-weight 5.0 --suffix reviewkd1

python train.py --model shufflev1 --teacher wrn-40-2 --teacher-weight checkpoints/cifar100_wrn-40-2__baseline1_best.pt --kd-loss-weight 5.0 --suffix reviewkd1

python train.py --model resnet8x4 --teacher resnet32x4 --teacher-weight checkpoints/cifar100_resnet32x4__baseline1_best.pt --kd-loss-weight 5.0 --suffix reviewkd1

python train.py --model shufflev1 --teacher resnet32x4 --teacher-weight checkpoints/cifar100_resnet32x4__baseline1_best.pt --kd-loss-weight 5.0 --suffix reviewkd1

python train.py --model shufflev2 --teacher resnet32x4 --teacher-weight checkpoints/cifar100_resnet32x4__baseline1_best.pt --kd-loss-weight 8.0 --suffix reviewkd1

python train.py --model resnet20 --teacher resnet56 --teacher-weight checkpoints/cifar100_resnet56__baseline1_best.pt --kd-loss-weight 0.6 --suffix reviewkd1

python train.py --model resnet32 --teacher resnet110 --teacher-weight checkpoints/cifar100_resnet110__baseline1_best.pt --kd-loss-weight 1.0 --suffix reviewkd1

