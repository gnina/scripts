#!/usr/bin/env python3

import argparse as ap
import numpy as np
import pandas as pd
import pickle

# Parse arguments
parser = ap.ArgumentParser()
parser.add_argument("files", nargs="+")
args = parser.parse_args()

# Open first file to get targets
print("Reading targets...", end="", flush=True)
targets = np.loadtxt(args.files[0], usecols=1, dtype="U4")
n_targets = len(targets)
print("done")

# Build dataframe
# Initializing the DF with a numpy array is essential for speed at assignment
print("Allocating DataFrame memory...", end="", flush=True)
# Protein sequence distance
df_dist = pd.DataFrame(index=targets, columns=targets, data=-1*np.ones((n_targets,n_targets)))
# Ligand similarity
df_lsim = pd.DataFrame(index=targets, columns=targets, data=-1*np.ones((n_targets,n_targets)))
print("done")

print("Merging data...", end="", flush=True)
for fname in args.files:
    target = np.loadtxt(fname, usecols=0, dtype="U4")[0]
    ctargets = np.loadtxt(fname, usecols=1, dtype="U4") # Can be removed if targets == ctargets all the time
    dist = np.loadtxt(fname, usecols=2)
    lsim = np.loadtxt(fname, usecols=3)
    
    if len(dist) == n_targets:
        df_dist.loc[target, ctargets] = dist
    if len(lsim) == n_targets:
        df_lsim.loc[target, ctargets] = lsim
print("done")

print(df_dist.values[:10,:10])
print(df_lsim.values[:10,:10])

# Check
print("Checking data...", end="", flush=True)
for i in range(n_targets):
    for j in range(n_targets):

        if df_dist.values[i,j] < 0:
            print("  Missing distance for", targets[i], targets[j])

        if df_lsim.values[i,j] < 0:
            print("  Missing ligand similarity for",targets[i],targets[j])

print("done")

print("Dumping pickle object...", end="", flush=True)
pickle.dump((df_dist.values, targets, df_lsim.values), open('matrix.pickle','wb'),-1)
print("done")