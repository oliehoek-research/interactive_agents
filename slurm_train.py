"""Use this script to launch experiments on a SLURM cluster.

This script does not need to be run within a singularity container,
and only imports modules from the python standard library.

This script accepts all of the arguments that "train_local.py" does,
except for the "--num-cpus" argument (use "--cpus-per-task" instead) and
the "--gpu" flag, since SLURM GPU allocation is not yet supported by our 
code.  This script also allows us to specify a Singulrity image in which to
run experiments, and specify the details of the SLURM resource allocation
for each job.

Any unrecognized keyword arguments will be passed to the
trainer clases (to support algorithm-specific arguments).

The script will run singularity locally (on the login node)
to setup the directory structure for the experiments, and then launch a
SLURM job array with a job for each configuration and seed.  

This script runs the "slurm_setup.py" locally to initialize the experiment
directories, and then launches "slurm_run.py" with "sbatch" to run each trial
as a separate SLURM job.
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
    parser.add_argument("--slurm-output", type=str, default=r"./results/slurm_output/%x_%a.out",
                        help="SLURM job name")
    parser.add_argument("--max-tasks", type=int, default=20,
                        help="maximum number of SLURM tasks allowed to run in parallel")

    parser.add_argument("-o", "--output-path", type=str, default="./results/debug",
                        help="directory in which we should save results (will be mounted in each Singularity container)")
    
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print data for every training iteration")
    parser.add_argument("--flush-secs", type=int, default=200,
                        help="number of seconds after which we should flush the training longs (default 200)")

    return parser.parse_known_args()


if __name__ == '__main__':
    args, unknown = parse_args()

    # Initialize experiment directories
    setup_command = [
        "singularity", 
        "exec",
        "--bind",
        f"{args.output_path}:/mnt/output",
        args.image, 
        "python3", 
        "slurm_setup.py",
        "--output-path",
        "/mnt/output"
    ]
    setup_command.extend(unknown)

    setup_process = subprocess.run(setup_command, stdout=subprocess.PIPE)
    paths = setup_process.stdout.decode("utf-8").splitlines()  # NOTE: Make sure to decode stdout before parsing it

    # Launch trials in SLURM
    run_command = [
        "singularity", 
        "exec",
        "--bind",
        f"{args.output_path}:/mnt/output",
        args.image, 
        "python3", 
        "slurm_run.py",
        "--flush-secs",
        args.flush_secs
    ]

    if args.verbose:
        run_command.append("verbose")

    run_command.extend(paths)

    # Join Singularity command into a single string
    run_command = [f'"{token}"' for token in run_command]
    run_command = " ".join(run_command)

    # Launch SLURM job array to run experiments
    slurm_command = ["sbatch"]
    slurm_command.extend([
        f"--partition={args.partition}",
        f"--qos={args.qos}",
        f"--time={args.time}",
        f"--cpus-per-task={args.cpus_per_task}",
        f"--mem-per-cpu={args.mem_per_cpu}",
        f"--job-name={args.job_name}",
        f"--output={args.slurm_output}"
    ])
    slurm_command.append(f"--array=0-{len(paths) - 1}%{args.max_tasks}")
    slurm_command.append("--wrap")
    slurm_command.append(run_command)

    subprocess.run(slurm_command)
