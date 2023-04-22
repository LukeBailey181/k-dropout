#!/bin/bash

# model
INPUT_DIM=$((1*32*32))
OUTPUT_DIM=10
HIDDEN_SIZE=1200
N_HIDDEN=2

# experiment
STANDARD_RUN="test_cifar_standard"
DROPOUT_RUN="test_cifar_dropout"

EPOCHS=1
LR=0.0005
P=0.5 
BATCH_SIZE=512

python train_net.py \
    --dataset_name cifar10 \
    --local_only \
    --preprocess_dataset \
    --batch_size $BATCH_SIZE \
    --input_size $INPUT_DIM \
    --hidden_size $HIDDEN_SIZE \
    --output_size $OUTPUT_DIM \
    --n_hidden $N_HIDDEN \
    --epochs $EPOCHS \
    --lr $LR \
    --p $P \
    --dropout_layer none \
    --num_workers 4 \
    --run_name {$DROPOUT_RUN}  


python train_net.py \
    --dataset_name cifar10 \
    --preprocess_dataset \
    --local_only \
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
    --num_workers 4 \
    --run_name {$STANDARD_RUN}  
