#!/usr/bin/env python3

import argparse as ap
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm

# Parse arguments
parser = ap.ArgumentParser()
parser.add_argument("files", nargs="+", type=str)
parser.add_argument("-out", "--output", type=str)
args = parser.parse_args()

# Open first file to get targets
print("Reading targets...", end="", flush=True)
targets = np.loadtxt(args.files[0], usecols=1, dtype="U4")
n_targets = len(targets)
print("done")

# Build DataFrame
# The DF are used to ensure that distances and ligand similarities are inserted at the correct place
# Initializing the DF with a numpy array is essential for speed at assignment
print("Allocating DataFrame memory...", end="", flush=True)
df_dist = pd.DataFrame(
    index=targets, columns=targets, data=-1 * np.ones((n_targets, n_targets))
)
df_lsim = pd.DataFrame(
    index=targets, columns=targets, data=-1 * np.ones((n_targets, n_targets))
)
print("done")

print("Merging data...", flush=True)
for fname in tqdm(args.files):
    target = np.loadtxt(fname, usecols=0, dtype="U4")[0]
    ctargets = np.loadtxt(fname, usecols=1, dtype="U4")
    dist = np.loadtxt(fname, usecols=2)
    lsim = np.loadtxt(fname, usecols=3)

    # Populate distance matrix
    if len(dist) == n_targets:
        df_dist.loc[target, ctargets] = dist
    else:
        print("  Invalid number of distances for {target}")

    # Populate ligand similarity matrix
    if len(lsim) == n_targets:
        df_lsim.loc[target, ctargets] = lsim
    else:
        print("  Invalid number of ligand similarities for {target}")

dist = df_dist.values
lsim = df_lsim.values

# Check properties
print("Checking matrix properties...", flush=True)
ddist, dlsim = np.diagonal(dist), np.diagonal(lsim)
assert int(round(np.sum(ddist))) == int(round(np.sum(ddist[ddist < 0])))
assert int(round(np.sum(dlsim[dlsim >= 0]))) - int(round(np.sum(dlsim[dlsim < 0]))) == n_targets
print("done")

# Set NaNs for compatibility with original implementation
dist[dist < 0] = np.nan
lsim[lsim < 0] = np.nan

print("Checking data...", flush=True)
rows, cols = np.where(np.isnan(dist))  # Invalid distances
for t1, t2 in zip(df_dist.index.values[rows], df_dist.columns.values[cols]):
    print(f"  Missing distance for {t1} {t2}")
rows, cols = np.where(np.isnan(lsim))  # Invalid ligand similarities
for t1, t2 in zip(df_dist.index.values[rows], df_dist.columns.values[cols]):
    print(f"  Missing ligand similarity for {t1} {t2}")

print(f"Dumping pickle object {args.output}...", end="", flush=True)
pickle.dump((dist, targets, lsim), open(f"{args.output}", "wb"), -1)
print("done")
