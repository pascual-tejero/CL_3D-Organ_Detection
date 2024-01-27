##>>>> active session
salloc -p lrz-dgx-a100-80x8 --ntasks=1 --gres=gpu:1 --time=72:00:00

salloc -p lrz-dgx-a100-80x8 --gres=gpu:1 --time=72:00:00 --mem-per-gpu 512000
salloc -p mcml-dgx-a100-40x8 --time 120:00:00 --gres gpu:1 -q mcml



## activate the env
ENV_name=msadet
BASE_DIR=/dss/dsshome1/0A/di93liq
ENV_DIR=ENVs_mori
DATA_DIR=/dss/dssmcmlfs01/pr62la/pr62la-dss-0002/OrganDetection/DATASET/

srun --pty --container-mounts=${DATA_DIR}:/mnt/data,${BASE_DIR}:/mnt/code \
--container-image=${BASE_DIR}/${ENV_DIR}/${ENV_name}.sqsh bash



cd /dss/dsshome1/0A/di93liq/transoar_def_detr
ENV_name=msadet
export TRANSOAR_DATA="/mnt/data/transoar_prep/dataset/"



# train with attn_fpn as backbone
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


