#!/bin/bash
#SBATCH --job-name=finetune-llama-70b  # Job name
#SBATCH --partition=i64m1tga800u       # Partition name
#SBATCH --nodes=3                      # Number of requested nodes
#SBATCH --ntasks-per-node=8            # Number of tasks per node
#SBATCH --cpus-per-task=8              # Number of CPUs per task
#SBATCH --gres=gpu:8                   # Number of GPUs per node
#SBATCH --time=6:00:00                 # Maximum runtime
#SBATCH --output=finetune-full-%j.out  # Standard output file
#SBATCH --error=finetune-full-%j.err   # Standard error file


module load cuda/12.2

source activate lit-gpt

srun python finetune/full.py --train_data_dir="data/train_watermarking" --val_data_dir="data/train_watermarking" --io.checkpoint_dir="/hpc2hdd/home/cyang662/pre-trained-models/Llama-2-70b-hf",  --out_dir="out/full-finetune/Llama-2-70b-watermarking" --devices=8 --num_nodes=3