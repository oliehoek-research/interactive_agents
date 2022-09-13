#!/usr/bin/bash
CONFIG_FILE=$1
OUTPUT_PATH=$2
FLAGS=""

NUM_TASKS=$(singularity exec singularity_image.sif python3 train_slurm.py ${CONFIG_FILE} -o ${OUTPUT_PATH} --setup ${FLAGS})
ARRAY_MAX=$((NUM_TASKS - 1))

sbatch --partition=debug \
    --time=1:00:00 \
    --cpus-per-task=1 \
    --job-name=Dummy-Test \
    --array=0-${ARRAY_MAX} \
    --wrap "singularity exec singularity_image.sif python3 train_slurm.py ${CONFIG_FILE} -o ${OUTPUT_PATH} ${FLAGS}"
