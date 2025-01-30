#!/bin/bash
#PBS -P iq24
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -q gpuvolta
#PBS -l mem=64GB
#PBS -l storage=gdata/iq24+scratch/iq24
#PBS -l walltime=12:00:00
#PBS -l jobfs=10GB

# fed_prompt fed_base base DFP_local
# gpuvolta dgxa100  fed_prompt base_10gloabl_50local_renew_train_different_lr_3
#MODEL_NAME=ViT_S_16 ResNet50
#FEATURE_TYPE=ViT_features  R50_features
# HIPT CLAM_SB TransMIL
# ACMIL adamw adam, lr =0.001 0.0002
# MODEL_NAME/FEATURE_TYPE -> MODEL_NAME_MIL -> MODEL_NAME_FED -> EXP_CODE -> top_k/G_EPOCHS

PROJECT_ID=iq24
MODEL_NAME=ResNet50
FEATURE_TYPE=R50_features
MODEL_NAME_MIL=ACMIL
MODEL_NAME_FED=fed_prompt
OPTIMIZER=adamw
OPTIMIZER_IMAGE=sgd
EXP_CODE=DFP_local
G_EPOCHS=10
N_PROMPTS=1
LOCAL_EPOCHS=50
REPEAT=3
LR=0.001
DATA_NAME=CAMELYON17
FT_ROOT=/g/data/$PROJECT_ID/CAMELYON17_patches/centers
CODE_ROOT=/scratch/iq24/cc0395/FedDDHist

cd $CODE_ROOT
source $HOME/ccenv/bin/activate
echo "Current Working Directory: $(pwd)"

# --heter_model \
python3 main.py \
--heter_model \
--feature_type $FEATURE_TYPE \
--ft_model $MODEL_NAME \
--mil_method $MODEL_NAME_MIL \
--fed_method $MODEL_NAME_FED \
--opt $OPTIMIZER \
--contrast_mu 2 \
--mu 0.01 \
--repeat $REPEAT \
--n_classes 4 \
--drop_out \
--lr $LR \
--pretrain_kd \
--syn_size 64 \
--global_epochs_dm 50 \
--ipc 10 \
--nps 100 \
--dc_iterations 1000 \
--image_lr 1.0 \
--image_opt $OPTIMIZER_IMAGE \
--B 8 \
--accumulate_grad_batches 1 \
--task $DATA_NAME \
--exp_code $EXP_CODE \
--global_epochs $G_EPOCHS \
--local_epochs $LOCAL_EPOCHS \
--bag_loss ce \
--inst_loss svm \
--results_dir $CODE_ROOT/exp \
--data_root_dir $FT_ROOT \
--prompt_lr 3e-4 \
--prompt_initialisation random \
--prompt_aggregation multiply \
--number_prompts $N_PROMPTS \
--key_prompt 4 \
--share_blocks 0 1 2 3 4 \
--share_blocks_g   5 6 \
--image_size 224 \
--renew_train \
--top_k 5000 \
#--debug

# --prompt_lr 3e-4 \
