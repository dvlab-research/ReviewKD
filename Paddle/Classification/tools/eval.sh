
# single card
# python3.7 tools/eval.py -c configs/r50_mv1_reviewkd.yaml -o pretrained_model="./ppcls"

# multi cards
python3.7 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir="./log_eval" tools/eval.py -c configs/r50_mv1_reviewkd.yaml -o pretrained_model="./output/DistillationModel/best_model/ppcls"
