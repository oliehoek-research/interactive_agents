import argparse

def parse_args():
    parser = argparse.ArgumentParser("Partial argument parsing test")

    parser.add_argument("arg1", type=str)
    parser.add_argument("--arg2", type=int, default=0)

    return parser.parse_known_args()


if __name__ == "__main__":
    args, rest = parse_args()

    print(rest)
