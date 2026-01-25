#!/bin/bash
#SBATCH --job-name=train_mdlm_2.5b
#SBATCH --output=train_mdlm_2.5b%j.out
#SBATCH --error=train_mdlm_2.5b%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=24:00:00
#SBATCH --mem=180GB

# Activate your venv
source /home/sagoyal/research/trm_vs_mdlm/.venv/bin/activate

# Already in the right directory, just run the command
accelerate launch --num_processes=4 --mixed_precision=bf16 train.py --output_dir outputs
