#!/bin/bash

# Print PyTorch and CUDA information
python -u -c 'import torch; print(torch.__version__); print(torch.cuda.device_count())'

# Path configurations
CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models

# Validate minimum required arguments
if [ $# -lt 5 ]; then
    echo "Error: Insufficient arguments!"
    echo "Usage: $0 MODE MODEL DATASET GPU_DEVICE SAVE_ID [BATCH_SIZE NEGATIVE_SAMPLE_SIZE HIDDEN_DIM GAMMA ALPHA LEARNING_RATE MAX_STEPS TEST_BATCH_SIZE REG P] [ADDITIONAL_FLAGS...]"
    echo ""
    echo "MODE options: train | valid | test"
    echo "Example: bash runs.sh train Rotate4D wn18 0 0 512 256 1000 12.0 1.0 0.0001 80000 8 0 2 --disable_adv"
    exit 1
fi

# Required parameters
MODE=$1
MODEL=$2
DATASET=$3
GPU_DEVICE=$4
SAVE_ID=$5

# Construct paths
FULL_DATA_PATH=$DATA_PATH/$DATASET
SAVE=$SAVE_PATH/"${MODEL}"_"${DATASET}"_"${SAVE_ID}"

# Training-specific parameters (with defaults)
BATCH_SIZE=${6:-512}
NEGATIVE_SAMPLE_SIZE=${7:-256}
HIDDEN_DIM=${8:-1000}
GAMMA=${9:-12.0}
ALPHA=${10:-1.0}
LEARNING_RATE=${11:-0.0001}
MAX_STEPS=${12:-80000}
TEST_BATCH_SIZE=${13:-8}
REG=${14:-0}
P=${15:-2}

# Additional flags (from position 16 onwards)
ADDITIONAL_FLAGS="${@:16}"

# Mode execution
case $MODE in
    train)
        echo "============================================"
        echo "Start Training..."
        echo "Model: $MODEL"
        echo "Dataset: $DATASET"
        echo "GPU Device: $GPU_DEVICE"
        echo "Save Path: $SAVE"
        echo "============================================"
        echo "Hyperparameters:"
        echo "  Batch Size: $BATCH_SIZE"
        echo "  Negative Sample Size: $NEGATIVE_SAMPLE_SIZE"
        echo "  Hidden Dimension: $HIDDEN_DIM"
        echo "  Gamma: $GAMMA"
        echo "  Alpha: $ALPHA"
        echo "  Learning Rate: $LEARNING_RATE"
        echo "  Max Steps: $MAX_STEPS"
        echo "  Test Batch Size: $TEST_BATCH_SIZE"
        echo "  Regularization: $REG"
        echo "  P-norm: $P"
        echo "  Additional Flags: $ADDITIONAL_FLAGS"
        echo "============================================"
        
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/runs.py \
            --do_train \
            --do_valid \
            --do_test \
            --data_path $FULL_DATA_PATH \
            --model $MODEL \
            -n $NEGATIVE_SAMPLE_SIZE \
            -b $BATCH_SIZE \
            -d $HIDDEN_DIM \
            -g $GAMMA \
            -a $ALPHA \
            -lr $LEARNING_RATE \
            --max_steps $MAX_STEPS \
            -save $SAVE \
            --test_batch_size $TEST_BATCH_SIZE \
            -reg $REG \
            -p $P \
            $ADDITIONAL_FLAGS
        ;;
        
    valid)
        echo "============================================"
        echo "Start Evaluation on Validation Data Set..."
        echo "Model Path: $SAVE"
        echo "============================================"
        
        if [ ! -d "$SAVE" ]; then
            echo "Error: Model directory '$SAVE' does not exist!"
            exit 1
        fi
        
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/runs.py \
            --do_valid \
            -init $SAVE
        ;;
        
    test)
        echo "============================================"
        echo "Start Evaluation on Test Data Set..."
        echo "Model Path: $SAVE"
        echo "============================================"
        
        if [ ! -d "$SAVE" ]; then
            echo "Error: Model directory '$SAVE' does not exist!"
            exit 1
        fi
        
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/runs.py \
            --do_test \
            -init $SAVE
        ;;
        
    *)
        echo "Error: Unknown MODE '$MODE'"
        echo "Valid modes are: train | valid | test"
        exit 1
        ;;
esac

echo "============================================"
echo "Done!"
echo "============================================"
