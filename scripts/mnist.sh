#!/bin/bash

# model
INPUT_DIM=784
OUTPUT_DIM=10
HIDDEN_SIZE=1200
N_HIDDEN=2

# experiment
RUN_NAME_PREFIX="mnist"
RESTARTS=1
EPOCHS=200
LR=0.0005
BATCH_SIZE=512
KS=(1 5 25 50 100 200 500 1000 2500 1000000)

for ((RESTART=1; RESTART<=$RESTARTS; RESTART++)); do
    # no dropout
    python train_net.py \
        --use_wandb \
        --run_name "${RUN_NAME_PREFIX}_nodropout_${RESTART}" \
        --dataset_name mnist \
        --preprocess_dataset \
        --batch_size $BATCH_SIZE \
        --input_size $INPUT_DIM \
        --hidden_size $HIDDEN_SIZE \
        --output_size $OUTPUT_DIM \
        --n_hidden $N_HIDDEN \
        --epochs $EPOCHS \
        --lr $LR \
        --dropout_layer none

    # sequential k dropout
    for K in "${KS[@]}"; do
        python train_net.py \
            --use_wandb \
            --run_name "${RUN_NAME_PREFIX}_sequential_${K}_${RESTART}" \
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
            --k $K
    done
done
