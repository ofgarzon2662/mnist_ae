#!/bin/bash
#SBATCH --job-name=mnist_ae_train
#SBATCH --partition=hotel-gpu            # or 'gpu-shared', 'gpu-debug', etc.
#SBATCH --QOS=hotel-gpu
#SBATH --account=use300
#SBATCH -N 1                       #Number of nodes
#SBATCH -n 1                       #Number of tasks per node
#SBATCH -G 1 
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out    # logs/mnist_ae_train_<jobid>.out
#SBATCH --error=logs/%x_%j.err     # (optional) separate stderr file

set -euo pipefail

echo "---- Loading modules ----"
module purge                           # good hygiene
module load python/3.11                # choose a Python that matches TSCC
module load cuda/12.4
module load cudnn/8.9.7                # if TSCC provides it separately

echo "---- Creating per-job venv ----"
python -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
python -m pip install --upgrade pip

echo "---- Installing GPU-enabled PyTorch ----"
python -m pip install torch==2.3.0+cu124 torchvision==0.18.0+cu124 \
      --extra-index-url https://download.pytorch.org/whl/cu124

echo "---- Installing your wheel ----"
python -m pip install $HOME/mnist_ae-0.0.1-py3-none-any.whl

echo "---- Starting training ----"
# Option A – if you added console_scripts = mnist_train=mnist_ae.mnist_training:main
mnist_train --epochs 5 --batch_size 128

# Option B – without the console script:
# python -m mnist_ae.mnist_training --epochs 5 --batch_size 128

echo "---- Job finished ----"