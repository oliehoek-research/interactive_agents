"""Use this script to launch experiments on a SLURM cluster.

This script does not need to be run within a singularity container,
and only imports modules from the python standard library.

This script accepts all of the arguments that "train_local.py" does,
except for the "--gpu" flag, since GPU allocation is not yet supported.
This script also allows us to specify a Singulrity image in which to
run experiments, and specify the details of the SLURM resource allocation
for each job.  Any unrecognized keyword arguments will be passed to the
trainer clases (to support algorithm-specific arguments).

The script will run singulraity locally (on the login node)
to setup the directory structure for the experiments, and then launch a
SLURM job array with a job for each configuration and seed.  This script
runs the "run_slurm.py" script, which should never be launched manually.
"""
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-i", "--image", type=str, default="./singularity_image.sif",
                        help="singularity image in which to run experiments")
    parser.add_argument("-p", "--partition", type=str, default="influence",
                        help="name of SLURM partition to use")
    parser.add_argument("-q", "--qos", type=str, default="short",
                        help="SLURM qos to use")
    parser.add_argument("-t", "--time", type=str, default="1:00:00",
                        help="SLURM time limit per job")
    parser.add_argument("-c", "--cpus-per-task", type=str, default="1",
                        help="CPUs per SLURM task")
    parser.add_argument("-m", "--mem-per-cpu", type=str, default="512M",
                        help="memory per SLURM CPU")
    parser.add_argument("--job-name", type=str, default="Ad-Hoc_Cooperation",
                        help="SLURM job name")
    
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unknown = parse_args()
    
    # Wrap python arguments in single quotes
    python_args = [f"'{arg}'" for arg in unknown]

    # Wrap Singularity image path in single quotes
    image = f"'{args.image}'"

    # Construct base python command
    run_command = ["singularity", "exec", image, "python3", "run_slurm.py"]
    run_command.extend(python_args)

    # Initialize directory structure and get the number of jobs to run
    setup_command = run_command + ["--setup"]
    setup_process = subprocess.run(setup_command, capture_output=True)
    num_tasks = int(setup_process.stdout)

    # Launch SLURM job array to run experiments
    slurm_command = ["sbatch"]
    slurm_command.extend([
        f"--partition={args.partition}",
        f"--qos={args.qos}",
        f"--time={args.time}",
        f"--cpus-per-task={args.cpus_per_task}",
        f"--mem-per-cpu={args.mem_per_cpu}",
        f"--job-name={args.job_name}"
    ])
    slurm_command.append(f"--array=0-{num_tasks - 1}")
    slurm_command.append("--wrap")
    
    train_command = " ".join(run_command)
    slurm_command.append(f'"{train_command}"')

    subprocess.run(slurm_command)
