#!/bin/bash

# model
INPUT_DIM=$((3*32*32))
OUTPUT_DIM=10
HIDDEN_SIZE=2000
N_HIDDEN=2

# experiment
RUN_NAME="pool_dropout_cifar_train"
MODEL_SAVE_DIR="./models/cifar10_ps_50.pt"

EPOCHS=300
LR=0.0005
P=0.5 
BATCH_SIZE=512
POOL_SIZE=50

python train_net.py \
    --dataset_name cifar10 \
    --preprocess_dataset \
    --batch_size $BATCH_SIZE \
    --input_size $INPUT_DIM \
    --hidden_size $HIDDEN_SIZE \
    --output_size $OUTPUT_DIM \
    --n_hidden $N_HIDDEN \
    --epochs $EPOCHS \
    --lr $LR \
    --p $P \
    --sync_over_model \
    --dropout_layer pool \
    --pool_size $POOL_SIZE\
    --num_workers 4 \
    --run_name $RUN_NAME \
    --model_save_path $MODEL_SAVE_DIR \
    --seed 282