#!/bin/bash

# SLURM Resource Parameters

#SBATCH -n 12                       # Number of cores
#SBATCH -N 1                        # Number of nodes always 1
#SBATCH -t 3-12:00 # D-HH:MM        # Time using the nodes
#SBATCH -p a100-gpu-shared               # Partition you submit to
#SBATCH --gres=gpu:2               # GPUs
#SBATCH --mem=32G                   # Memory you need
#SBATCH --job-name=SmallFlowFormerFinetune      # Job name
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
#SBATCH --mail-type=ALL             # Type of notification BEGIN/FAIL/END/ALL
#SBATCH --mail-user=satanama.ring@gmail.com
# Executable
EXE=/bin/bash

singularity exec --nv --bind /data2/datasets/wenshanw/tartan_data:/zihao/datasets:ro,/data2/datasets/yuhengq/zihao/FlowFormer:/zihao/FlowFormer /data2/datasets/yuhengq/zihao/flowformer.sif bash /zihao/FlowFormer/script.sh