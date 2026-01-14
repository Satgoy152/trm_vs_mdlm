#!/bin/bash
#SBATCH --job-name=train_trm_2.5b
#SBATCH --output=train_trm_2.5b%j.out
#SBATCH --error=train_trm_2.5b%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=24:00:00
#SBATCH --mem=180GB

# Activate your venv
source /home/sagoyal/research/trm_vs_mdlm/.venv/bin/activate

# Already in the right directory, just run the command
accelerate launch --num_processes=4 --mixed_precision=bf16 train.py --method trm --output_dir outputs
