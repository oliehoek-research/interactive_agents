import argparse
import os.path
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--list", action="store_true",
                        help="lists the paths to be loaded, one path per line")
    parser.add_argument("--path", type=str,
                        help="the path to be loaded")

    return parser.parse_args()


BASE_PATH = "./experiment"
SEEDS = [334, 2144, 5433, 6823]

if __name__ == '__main__':
    args = parse_args()

    if args.list:
        for seed in SEEDS:
            path = os.path.join(BASE_PATH, f"seed_{seed}")
            print(path)
    elif args.path is not None:
        print(f"loading path: {args.path}")
    else:
        setup_command = ["python3", "subprocess_test.py", "--list"]
        setup_process = subprocess.run(setup_command, stdout=subprocess.PIPE)

        # NOTE: "stdout" is a bytestring, which we need to decode before treating it as a Python string
        paths = setup_process.stdout.decode("utf-8").splitlines()
        for path in paths:
            command = ["python3", "subprocess_test.py", "--path", path]
            subprocess.run(command)
