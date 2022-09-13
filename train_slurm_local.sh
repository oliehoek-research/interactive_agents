#!/usr/bin/bash
IMAGE=$1
CONFIG_FILE=$2
OUTPUT_PATH=$3
FLAGS=""

NUM_TASKS=$(singularity exec ${IMAGE} python3 train_slurm.py ${CONFIG_FILE} -o ${OUTPUT_PATH} --setup ${FLAGS})
ARRAY_MAX=$((NUM_TASKS - 1))

sbatch --partition=debug \
    --time=1:00:00 \
    --cpus-per-task=1 \
    --job-name=Dummy-Test \
    --array=0-${ARRAY_MAX} \
    --wrap "singularity exec ${IMAGE} python3 train_slurm.py ${CONFIG_FILE} -o ${OUTPUT_PATH} ${FLAGS}"
