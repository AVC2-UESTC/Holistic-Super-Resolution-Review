#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
PORT=${PORT:-4811}

# usage
if [ $# -lt 2 ] ;then
    echo "usage:"
    echo "./scripts/dist_train.sh [number of gpu] [path to option file]"
    exit
fi

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    basicsr/train.py -opt $CONFIG --launcher pytorch ${@:3}

# ##### 
# GPUS=$1
# CONFIG=$2
# PORT=${PORT:-4111}

# # usage
# if [ $# -lt 2 ] ;then
#     echo "usage:"
#     echo "./scripts/dist_train.sh [number of gpu] [path to option file]"
#     exit
# fi

# PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
# CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
#     basicsr/train.py --opt options/train_dwten_scratch_x2.yml --launcher pytorch ${@:3}