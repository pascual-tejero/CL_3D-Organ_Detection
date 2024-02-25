##>>>> active session
salloc -p lrz-dgx-a100-80x8 --ntasks=1 --gres=gpu:1 --time=72:00:00
salloc -p lrz-dgx-a100-80x8 --gres=gpu:1 --time=72:00:00 --mem-per-gpu 512000 #

salloc -p mcml-dgx-a100-40x8 --time 96:00:00 --gres gpu:1 -q mcml



## activate the env
ENV_name=msadet
BASE_DIR=/dss/dsshome1/01/ge32fix2
ENV_DIR=ENVS
DATA_DIR=/dss/dsshome1/01/ge32fix2/DATASETS/

srun --pty --container-mounts=${DATA_DIR}:/mnt/data,${BASE_DIR}:/mnt/code \
--container-image=${BASE_DIR}/${ENV_DIR}/${ENV_name}.sqsh bash


*(code_organ_DETR)*
cd /mnt/code
ENV_name=msadet
export TRANSOAR_DATA=/mnt/data/



# train with attn_fpn as backbone
/opt/conda/envs/${ENV_name}/bin/python scripts/train.py --config CL_methods/msa_def_detr_CL_ANCL
/opt/conda/envs/${ENV_name}/bin/python scripts/train.py --config attn_fpn_def_detr_abdomen_full
/opt/conda/envs/${ENV_name}/bin/python scripts/train.py --config attn_fpn_def_detr_abdomen



# train with msavit as backbone
/opt/conda/envs/${ENV_name}/bin/python scripts/train.py --config msa_def_detr_abdomen_full 
/opt/conda/envs/${ENV_name}/bin/python scripts/train.py --config msa_def_detr_amos_full 




--config contrast_resnet_def_detr_abdomen



# test 
OCNFIG=def_detr_abdomenCT1k
msa_def_detr_abdomen


/opt/conda/envs/${ENV_name}/bin/python scripts/test.py --run ${OCNFIG}


# Login to LRZ
ssh login.ai.lrz.de -l ge32fix2

# Example copy dataset to LRZ from workstation
scp -r test_dataset ge32fix2@login.ai.lrz.de:/dss/dsshome1/01/ge32fix2/DATASET

# Quit environment
exit

# Tasks submitted
squeue -u ge32fix2 # My tasks
squeue # All tasks from users