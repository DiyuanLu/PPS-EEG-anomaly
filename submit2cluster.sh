#!/usr/bin/env sh

#SBATCH --partition sleuths
##SBATCH --nodelist jetski
##SBATCH --nodelist speedboat
#SBATCH --nodes 1
#SBATCH --time 4000
#SBATCH --mincpus 2
#SBATCH --gres gpu:1

srun python3 train_aae.py "$@"


#command line: "sbatch --job-name 32141 --output %N_%j.log --error %N_%j.log submit2cluster.sh"