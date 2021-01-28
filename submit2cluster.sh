#!/usr/bin/env sh

#SBATCH --partition sleuths
##SBATCH --reservation triesch-shared
##SBATCH --nodelist jetski
##SBATCH --nodelist speedboat
#SBATCH --nodes 1
#SBATCH --mem 15GB
#SBATCH --mincpus 2
##SBATCH --gres gpu:1


srun python3 train_aae.py
