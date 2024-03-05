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

srun python finetune/full.py --io.train_data_dir="../data/train_watermarking" --io.val_data_dir="../data/train_watermarking" --io.checkpoint_dir="/home/yangchenyu/pre-trained-models/Llama-2-7b-hf"  --io.out_dir="out/full-finetune/Llama-2-7b-watermarking" --devices=8 --num_nodes=3