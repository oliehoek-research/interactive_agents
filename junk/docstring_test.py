'''
This script just tests that ArgParse can load the default docstring.

If this message is displayed, the test has succeeded!
'''

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-t", "--test", type=str, default="test_value", 
        help="dummy argument for testing")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
