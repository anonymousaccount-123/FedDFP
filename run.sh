#!/bin/bash
#PBS -P iq24
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -q gpuvolta
#PBS -l mem=64GB
#PBS -l storage=gdata/iq24+scratch/iq24
#PBS -l walltime=48:00:00
#PBS -l jobfs=10GB

# fed_prompt fed_base base DFP_local
# gpuvolta dgxa100  CAMELYON16_IMAGE fed_prompt base_10gloabl_50local_renew_train_different_lr_3
#MODEL_NAME=ViT_S_16 ResNet50
#FEATURE_TYPE=ViT_features  R50_features
# HIPT CLAM_SB TransMIL
# ACMIL adamw adam, lr =0.001 0.0002
# MODEL_NAME/FEATURE_TYPE -> MODEL_NAME_MIL -> MODEL_NAME_FED -> EXP_CODE -> top_k/G_EPOCHS

PROJECT_ID=iq24
MODEL_NAME=ViT_S_16
FEATURE_TYPE=ViT_features
MODEL_NAME_MIL=ACMIL
MODEL_NAME_FED=fed_prompt
OPTIMIZER=adamw
OPTIMIZER_IMAGE=sgd
EXP_CODE=base
LOCAL_EPOCHS=50
G_EPOCHS=10
N_PROMPTS=1
REPEAT=3
LR=0.001
DATA_NAME=CAMELYON16 #CAMELYON16_IMAGE CAMELYON17
FT_ROOT=/g/data/$PROJECT_ID/CAMELYON16_patches
CODE_ROOT=/scratch/iq24/cc0395/FedDDHist

cd $CODE_ROOT
source /g/data/$PROJECT_ID/mmcv_env/bin/activate
echo "Current Working Directory: $(pwd)"

# --heter_model \
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
--prompt_lr 3e-3 \
--prompt_initialisation random \
--prompt_aggregation multiply \
--number_prompts $N_PROMPTS \
--key_prompt 4 \
--share_blocks 0 1 2 3 4 \
--share_blocks_g   5 6 \
--image_size 224 \
--renew_train \
#--debug
#--top_k 5000 \
#--debug
# --prompt_lr 3e-4 \

#cmake -S . -B build \
#-DPugiXML_INCLUDE_DIR="$PATH_ASAP_EXTLIBS/pugixml/install/usr/local/include/" \
#-DOPENSLIDE_INCLUDE_DIR="$PATH_ASAP_EXTLIBS/openslide/install/usr/local/include/openslide" \
#-DOPENSLIDE_LIBRARY="$PATH_ASAP_EXTLIBS/openslide/install/usr/local/lib/libopenslide.so.0.4.1" \
#-DOpenCV_DIR="$PATH_ASAP_EXTLIBS/opencv-4.x/build" \
#-DDCMTK_DIR="$PATH_ASAP_EXTLIBS/dcmtk-DCMTK-3.6.7/build" \
#-DDCMTKJPEG_INCLUDE_DIR="$PATH_ASAP_EXTLIBS/dcmtk-DCMTK-3.6.7/install/usr/local/include" \
#-DDCMTKJPEG_LIBRARY="$PATH_ASAP_EXTLIBS/dcmtk-DCMTK-3.6.7/install/usr/local/lib64/libijg8.a" \
#-DBUILD_EXECUTABLES=ON \
#-DBUILD_ASAP=ON \
#-DBUILD_IMAGEPROCESSING=ON \
#-DPACKAGE_ON_INSTALL=ON  \
#-DBUILD_MULTIRESOLUTIONIMAGEINTERFACE_VSI_SUPPORT=ON \
#-DBUILD_MULTIRESOLUTIONIMAGEINTERFACE_DICOM_SUPPORT=ON \
#-DWRAP_MULTIRESOLUTIONIMAGEINTERFACE_PYTHON=ON \
##-DPYTHON_NUMPY_INCLUDE_DIR='/g/data/iq24/mmcv_env/lib/python3.9/site-packages/numpy/_core/include'\
#-DPYTHON_LIBRARY=$CONDA_PREFIX/lib/libpython3.9.so