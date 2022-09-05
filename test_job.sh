#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=influence 
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --job-name=Dummy-Test

singularity exec singularity_image.sif test_script.py


#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --mem=16G  # Is this memory per node?
#SBATCH --output=memorygame_%a.out
#SBATCH --array=1-90  # Why is this fixed rather than being adaptive?
#SBATCH --job-name=MemoryGame-Test

# TODO: Break this command across multiple lines so we can read it
# TODO: Where are configs read from/written to when running on the cluster?
# TODO: We won't need the triton command
singularity exec $WRKDIR/torchenv.sif python3 train.py \
    -f ./experiments/r2d2_memory/config_r2d2_memory_experiment`expr $SLURM_ARRAY_TASK_ID % 18`.yaml \
    -o ./triton_multiagent_results/r2d2_memorygame_lengthVScues`expr $SLURM_ARRAY_TASK_ID % 18`/ \
    --seeds `expr $SLURM_ARRAY_TASK_ID % 5` \
    --triton