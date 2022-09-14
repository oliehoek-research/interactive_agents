import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--setup", action="store_true",
                        help="prints the number of tasks to run")
    parser.add_argument("--task", type=int,
                        help="the current task ID")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.setup:
        print(2)
    elif args.task is not None:
        print(f"running task {args.task}")
    else:
        setup_command = ["python3", "subprocess_test.py", "--setup"]
        setup_process = subprocess.run(setup_command, capture_output=subprocess.PIPE)
        
        num_tasks = int(setup_process.stdout)
        for task in range(num_tasks):
            command = ["python3", "subprocess_test.py", "--task", str(task)]
            subprocess.run(command)
