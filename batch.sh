#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kitty.fung@magd.ox.ac.uk
python3 Rmax-arch1_batch.py
