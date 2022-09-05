#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --job-name=Dummy-Test
#SBATCH --array=0-4

singularity exec singularity_image.sif python3 test_script.py -j `expr ${SLURM_ARRAY_TASK_ID}`