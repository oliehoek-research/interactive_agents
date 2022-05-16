#!/usr/bin/env python3
"""
Script for generating plots from data stored in RLLib-generated CSV files
"""

import argparse
from collections import namedtuple
import os
import pandas  # NOTE: Pandas may not be an option, as it seems to throw a security error - even if this script is not itself a vulnerability.

import matplotlib.pyplot as plot
import matplotlib.patches as patches
import matplotlib.cm as colors
import numpy as np
import scipy
import scipy.stats


def parse_args():
    parser = argparse.ArgumentParser("Generates a plot of a scalar-valued time series from a set of experiments.")

    parser.add_argument("experiments", type=str, nargs="*", 
                        help="labels and directories of experiments to plot (label1 dir1 label2 dir2 ...)")
    parser.add_argument("--output", default="mean_return", type=str,
                        help="path to the image file where the plot will be saved")
    parser.add_argument("--x-axis", default="total_episodes", type=str, 
                        help="column name for x-axis values")
    parser.add_argument("--y-axis", default="eval/mean_reward", type=str, 
                        help="column name for y-axis values")
    parser.add_argument("--x-label", default="time steps", type=str,
                        help="label for the x-axis")
    parser.add_argument("--y-label", default="mean episode return", type=str, 
                        help="label for the y-axis")
    parser.add_argument("--title", default="Mean Episode Return", type=str,
                        help="title for the plot to be generated")
    parser.add_argument("--errors", default="range", type=str,
                        help="error values to plot as shaded regions \{'range', 'deviation', 'error', 'None'\}")
    parser.add_argument("--incomplete", default="truncate", type=str,
                        help="how to handle incomplete runs \{'ignore', 'truncate'\}")

    return parser.parse_args()


def load_experiments(args):
    if len(args) % 2 != 0:  # NOTE: Need an even number of items (label experiment pairs)
        raise ValueError("Must provide a label for each experiment")

    experiments = dict()  # NOTE: Results for each labeled experiment

    for index in range(0, len(args), 2):  # NOTE: Steps through each label-experiment pair
        directory = args[index + 1]  # NOTE: Second item is a directory of experimental results
        runs = []  # NOTE: These are individual random seeds

        if not os.path.isdir(directory):
            raise Exception(f"Experiment directory {directory} does not exist")

        for obj in os.listdir(directory):  # NOTE: Grabs all sub-directories, regardless of their name
            path = os.path.join(directory, obj)

            if os.path.isdir(path):
                # NOTE: Loads the entire table with all columns - could be slow
                data = pandas.read_csv(os.path.join(path, "results.csv"))  # NOTE: Change this to 'results.csv' to work with our data

                # Filter out empy data series
                if data.shape[0] > 0:
                    runs.append(data)
        
        experiments[args[index]] = runs  # NOTE: Stores each table, for each seed, under the given label of the experiment
    
    return experiments


if __name__ == "__main__":
    args = parse_args()

    # Load experiment data
    experiments = load_experiments(args.experiments)  # NOTE: What does the resulting data look like?

    # Plot results
    color_map = colors.get_cmap("tab20").colors  # NOTE: Just a list of colors, but organized as pairs shades of the same color
    legend_entries = []  # NOTE: Pretty straightforward, just a list of items that will be added to the plot legend
    y_min = np.infty  # NOTE: tracks the range of the scalars to construct a suitable y-axis
    y_max = -np.infty

    plot.clf()  # NOTE: Matplotlib requires we clear the plot before drawing anything - needs to come before the loop (which draws each series)

    for index, (label, runs) in enumerate(experiments.items()):  # NOTE: Iterate over individual experiments - these will be the individual data series
        if len(runs) > 0:  # NOTE: Just ignore empty runs - incomplete runs can be handled in multiple ways

            lengths = [len(run) for run in runs]  # NOTE: A 'run' is actually a pandas dataframe, what does 'len' return for this?

            if "ignore" == args.incomplete:  # NOTE: This mode just removes incomplete runs
                # Compute the maximum length over runs
                max_length = max(lengths)  # NOTE: This looks like a bug! (should maximize not minimize)

                # Remove incomplete sequences
                runs = [run for run in runs if len(run) == max_length]  # NOTE: Is this safe in python?  Has 'runs' on both sides of the assignment

                # Define x-axis
                x_axis = runs[0][args.x_axis]  # NOTE: Seems to let Matplotlib handle formatting of the x-axis

                # Construct data series and compute means
                series = [run[args.y_axis] for run in runs]  # NOTE: Here we are pulling out the individual 
            else:  # NOTE: This mode truncates all runs to be the same length as the shortest run
                # Compute the minimum length over runs
                min_length = min(lengths)  # NOTE: This part is actually right, might have been a copy-paste bug

                # Define x-axis
                x_axis = runs[0][args.x_axis][0:min_length]  # NOTE: Now we have to truncate the x-axis

                # Construct data series and compute means
                series = [run[args.y_axis][0:min_length] for run in runs]  # NOTE: Pulls out and truncates the individual time series

            # Print run information
            print(f"\n\nExperiment: {label}")
            print(f"    data keys: {runs[0].keys()}")

            # Convert series data to a single numpy array
            series = np.asarray(series, dtype=np.float32)  # NOTE: Has to stack the list of arrays into a single array
            means = np.mean(series, axis=0)  # NOTE: Compute the mean over valide random seeds

            # Update ranges  # NOTE: Maintains a running range for the y-axis (can only expand as new data is loaded)
            y_min = min(y_min, np.min(series))
            y_max = max(y_max, np.max(series))

            # Compute error bars
            if "range" == args.errors:
                upper = np.max(series, axis=0)
                lower = np.min(series, axis=0)
            elif "deviation" == args.errors:
                std = np.std(series, axis=0, ddof=1)
                upper = means + std
                lower = means - std
            elif "error" == args.errors:  # NOTE: Here error refers the standard error, or the standard deviation divided by the root of the sample size
                error = scipy.stats.sem(series, axis=0, ddof=1)
                upper = means + error
                lower = means - error
            else:  # NOTE: Invalid mode string prints no error bars
                upper = means
                lower = means

            # Plot series
            plot.plot(x_axis, means, color=color_map[2 * index], alpha=1.0)  # NOTE: Plots the main sequence using the base color
            plot.fill_between(x_axis, lower, upper, color=color_map[2 * index + 1], alpha=0.3) # NOTE: Plots the error bars with a transparent shade of the base color

        # Add a legend entry even if there were no non-empty data series
        legend_entries.append(patches.Patch(color=color_map[2 * index], label=label))  # NOTE: Legend entries are just Patch objects defining a color and a text label - no direct link to the data

    # Set ranges  # NOTE: Just a mechanism for generating nice-looking y-axes
    if y_min > y_max:  # No data, set an arbitrary range
        y_min = 0.0
        y_max = 100.0
    elif 0.0 == y_min and 0.0 == y_max:  # All data is zero, set and arbitrary range
        y_min = -100.0
        y_max = 100.0
    elif y_min >= 0.0:  # All values positive, set range from 0 to 120% of max  # NOTE: Can change this to non-negative, won't have an effect but makes it more clear
        y_min = 0.0
        y_max *= 1.2
    elif y_max <= 0.0:  # All values negative, set range from 120% of min to 0  # NOTE: Likewise
        y_min *= 1.2
        y_max = 0.0
    else:  # Both positive and negative values, expand range by 20%
        y_min *= 1.2
        y_max *= 1.2

    # Create plot
    plot.legend(handles=legend_entries)  # NOTE: Add legend to plot
    plot.title(args.title)
    plot.xlabel(args.x_label)
    plot.ylabel(args.y_label)
    plot.ylim(bottom=y_min, top=y_max)  # NOTE: Sets the range of the y-axis
    plot.savefig(args.output, bbox_inches="tight")  # NOTE: Doesn't display the plot, just saves it - 'tight' seems to minimize whitespace (good for publications)
