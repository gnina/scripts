#!/usr/bin/env python3

'''Combine the output of compute_rows.py into a pickle file for clustering.py'''

import pickle, sys, collections
import numpy as np

target_names = []
targets = dict() # name to index
values = collections.defaultdict(dict) # indexed by row name, col name

for fname in sys.argv[1:]:
    for line in open(fname):
        (t1,t2,dist,lsim) = line.split()
        dist = float(dist)
        if t2 not in targets:
            targets[t2] = len(target_names)
            target_names.append(t2)
        values[t1][t2] = (dist,lsim)
        
        
#must have fully filled out matrix
l = len(target_names)
m = np.empty((l,l))
lm = np.empty((l,l))
m[:] = np.NAN
lm[:] = np.NAN

for t1 in values.keys():
    for t2 in values[t1].keys():
        i = targets[t1]
        j = targets[t2]
        m[i][j] = values[t1][t2][0]
        lm[i][j] = values[t1][t2][1]
    
#check throws a key error if a key is missing in targets
#      or prints the sentence if NAN is present
for i in range(l):
    for j in range(l):
        if not np.isfinite(m[i][j]):
            print("Missing distance for",targets[i],targets[j])

        if not np.isfinite(lm[i][j]):
            print("Missing ligand_sim for",targets[i],targets[j])

    
pickle.dump((m, target_names, lm), open('matrix.pickle','w'),-1)
