#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=influence 
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --job-name=Dummy-Test

singularity exec singularity_image.sif test_script.py