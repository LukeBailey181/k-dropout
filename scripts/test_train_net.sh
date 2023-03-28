#!/bin/bash

# model
INPUT_DIM=784
OUTPUT_DIM=10
HIDDEN_SIZE=1200
N_HIDDEN=2

# experiment
RUN_NAME_PREFIX="mnist"
RESTARTS=1
EPOCHS=1
LR=0.0005
BATCH_SIZE=512

python train_net.py \
    --local_only \
    --dataset_name mnist \
    --preprocess_dataset \
    --batch_size $BATCH_SIZE \
    --input_size $INPUT_DIM \
    --hidden_size $HIDDEN_SIZE \
    --output_size $OUTPUT_DIM \
    --n_hidden $N_HIDDEN \
    --epochs $EPOCHS \
    --lr $LR \
    --dropout_layer sequential \
    --m 2 \
    --num_workers 0 \
    --k 1 

python train_net.py \
    --local_only \
    --dataset_name mnist \
    --preprocess_dataset \
    --batch_size $BATCH_SIZE \
    --input_size $INPUT_DIM \
    --hidden_size $HIDDEN_SIZE \
    --output_size $OUTPUT_DIM \
    --n_hidden $N_HIDDEN \
    --epochs $EPOCHS \
    --lr $LR \
    --dropout_layer pool \
    --m 2 \
    --num_workers 0 \
    --pool_size 100 \
    --k 1 
