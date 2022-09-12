#!/usr/bin/env python3
"""Expands a grid-search configuration into a set of specifc configurations.

Takes a single .yaml configuration file with "grid_search" keys defining
the hyperparameters to be tuned, and the list of possible values they can
take.  Generates a set of fully-specified configurations, and saves them
as individual .yaml files in the given directory.
"""
import argparse

from interactive_agents import grid_search, load_configs  # NOTE: Move these back to the "util" package so they can be loaded with minimal dependencies
 
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-f", "--config-file", type=str,
                        help="grid-search config file from which specific configs will be generated")
    parser.add_argument("-o", "--output-path", type=str, default="./experiments",
                        help="directory in which we should save experiment configs") 

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    experiments = load_configs(args.config_file)

    # NOTE: This isn't strictly necessary
    assert len(experiments) == 1, f"'{args.config_file}' contains more than one grid-search config"

    for base_name, base_config in experiments:
        variations = grid_search(base_name, base_config)

        if len(variations) > 0:
            path = os.path.join(args.output_path, base_name)
            if not os.path.exists(path):
                os.makedirs(path)

            idx = 0
            for name, config in variations.items():   
                name = name.replace("=", "").replace("," , "_")
                config_path = os.path.join(path, f"config_{base_name}_experiment{idx}.yaml")
                idx += 1
            
                with open(config_path, 'w') as config_file:
                    yaml.dump({name: config}, config_file)
    
    print("Files have been generated.")
