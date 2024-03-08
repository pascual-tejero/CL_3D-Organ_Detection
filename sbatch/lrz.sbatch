##>>>> active session
salloc -p lrz-dgx-a100-80x8 --ntasks=1 --gres=gpu:1 --time=72:00:00
salloc -p mcml-dgx-a100-40x8 --time 96:00:00 --gres gpu:1 -q mcml



## activate the env
ENV_name=msadet
BASE_DIR=/dss/dsshome1/01/ge32fix2
ENV_DIR=ENVS
DATA_DIR=/dss/dsshome1/01/ge32fix2/DATASETS/

srun --pty --container-mounts=${DATA_DIR}:/mnt/data,${BASE_DIR}:/mnt/code \
--container-image=${BASE_DIR}/${ENV_DIR}/${ENV_name}.sqsh bash


cd /mnt/code/OrganDETR
ENV_name=msadet
export TRANSOAR_DATA=/mnt/data/



# train
/opt/conda/envs/${ENV_name}/bin/python scripts/train.py --config CL_methods/ABDOMEN-CT-1K_WORD/msa_def_detr_ABDOMEN-CT-1K
/opt/conda/envs/${ENV_name}/bin/python scripts/train.py --config CL_methods/ABDOMEN-CT-1K_WORD/msa_def_detr_WORD
/opt/conda/envs/${ENV_name}/bin/python scripts/train.py --config CL_methods/ABDOMEN-CT-1K_WORD/msa_def_detr_ANCL


# test
/opt/conda/envs/${ENV_name}/bin/python scripts/test.py --run msa_def_detr_ABDOMEN-CT-1K_dataset
/opt/conda/envs/${ENV_name}/bin/python scripts/test.py --run msa_def_detr_WORD_dataset
/opt/conda/envs/${ENV_name}/bin/python scripts/test.py --run msa_def_detr_ANCL_ABDOMEN-CT-1K_WORD_datasets



# Login to LRZ
ssh login.ai.lrz.de -l ge32fix2

# Example copy dataset to LRZ from workstation
scp -r test_dataset ge32fix2@login.ai.lrz.de:/dss/dsshome1/01/ge32fix2/DATASET

# Quit environment
exit

# Tasks submitted
squeue -u ge32fix2 # My tasks
squeue # All tasks from users