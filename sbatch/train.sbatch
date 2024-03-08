#!/bin/bash
#SBATCH -J test
#SBATCH -N 1
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:00:10
#SBATCH -o test_%x.%j.%N.out
#SBATCH -e test_%x.%j.%N.err


BASE_DIR=/dss/dsshome1/0A/di93liq
CODE_DIR=Dementia_Cond_Diffusion
ENV_DIR=ENVs_mori

ENV_name=torch18_docker

srun --mpi=pmi2 \
--container-mounts=${BASE_DIR}/myData:/mnt/data,${BASE_DIR}/${CODE_DIR}:/mnt/code  \
--container-image=${BASE_DIR}/${ENV_DIR}/ai-med+${ENV_name}+latest.sqsh \
--container-workdir=/mnt/code \
/opt/conda/envs/${ENV_name}/bin/python Mori_run.py --base_dir ${BASE_DIR}/${CODE_DIR}


#### sbatch train.sbatch 