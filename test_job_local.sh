FLAG="--setup --flag"

NUM_TASKS=$(singularity exec singularity_image.sif python3 test_script.py $FLAG)
echo $NUM_TASKS

# ARRAY_MAX=$((NUM_TASKS - 1))

# sbatch --partition=debug \
#     --time=1:00:00 \
#     --cpus-per-task=1 \
#     --job-name=Dummy-Test \
#     --array=0-${ARRAY_MAX} \
#     --wrap "singularity exec singularity_image.sif python3 test_script.py"
