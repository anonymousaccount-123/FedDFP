#!/bin/bash
#PBS -P iq24
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -q gpuvolta
#PBS -l mem=64GB
#PBS -l storage=gdata/iq24+scratch/iq24
#PBS -l walltime=24:00:00
#PBS -l jobfs=10GB

# fed_prompt fed_base base DFP_local
# gpuvolta dgxa100  CAMELYON16_IMAGE fed_prompt base_10gloabl_50local_renew_train_different_lr_3
#MODEL_NAME=ViT_S_16 ResNet50
#FEATURE_TYPE=ViT_features  R50_features
# HIPT CLAM_SB TransMIL
# MODEL_NAME/FEATURE_TYPE -> MODEL_NAME_MIL -> MODEL_NAME_FED -> EXP_CODE -> top_k/G_EPOCHS/debug

PROJECT_ID=iq24
MODEL_NAME=ResNet50
FEATURE_TYPE=R50_features
MODEL_NAME_MIL=CLAM_SB
MODEL_NAME_FED=fed_prompt
EXP_CODE=heter_model
G_EPOCHS=10
N_PROMPTS=10
LOCAL_EPOCHS=50
REPEAT=1
OPTIMIZER=adam
OPTIMIZER_IMAGE=adam
DATA_NAME=IDH
FT_ROOT=/g/data/$PROJECT_ID/IDH
CODE_ROOT=/scratch/iq24/cc0395/FedDDHist

cd $CODE_ROOT
source $HOME/ccenv/bin/activate
echo "Current Working Directory: $(pwd)"

#--heter_model \
python3 main.py \
--feature_type $FEATURE_TYPE \
--ft_model $MODEL_NAME \
--mil_method $MODEL_NAME_MIL \
--fed_method $MODEL_NAME_FED \
--opt $OPTIMIZER \
--contrast_mu 2 \
--mu 0.01 \
--repeat $REPEAT \
--n_classes 2 \
--drop_out \
--lr 2e-4 \
--pretrain_kd \
--syn_size 64 \
--global_epochs_dm 50 \
--kd_iters 10 \
--ipc 10 \
--nps 100 \
--dc_iterations 1000 \
--image_lr 1.0 \
--image_opt $OPTIMIZER_IMAGE \
--B 1 \
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
--renew_train \
#--top_k 5000 \
#--debug
