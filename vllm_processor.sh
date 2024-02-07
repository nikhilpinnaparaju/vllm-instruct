#!/bin/bash
#SBATCH --account="stablegpt"
#SBATCH --job-name=vllm-instruct
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=vllm_prompt/%x_%j.out
#SBATCH --error=vllm_prompt/%x_%j.err

input_file=$1

source /etc/profile.d/modules.sh
module load openmpi cuda/12.1

conda activate instruct

python mixtral-vllm.py $input_file