#!/bin/bash
#SBATCH --array=1-10
#SBATCH --time=1:00:00
#SBATCH --partition=influence 
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=300M
#SBATCH --output=./results/debug/slurm/dummy_%a.out
#SBATCH --job-name=Dummy-Test

singularity exec interactive_agents_singularity.sif dummy_train.py \
    -f ./experiments/proof_of_concept/r2d2_self_play_coordination.yaml \
    -o results/debug/slurm/dummy \
    -s ${SLURM_ARRAY_TASK_ID}
