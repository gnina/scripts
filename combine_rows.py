#!/usr/bin/env python

'''Combine the output of compute_rows.py into a pickle file for clustering.py'''

import cPickle, sys, collections
import numpy as np

target_names = []
targets = dict() # name to index
values = collections.defaultdict(dict) # indexed by row name, col name

for fname in sys.argv[1:]:
    for line in open(fname):
        (t1,t2,dist) = line.split()
        dist = float(dist)
        if t2 not in targets:
            targets[t2] = len(target_names)
            target_names.append(t2)
        values[t1][t2] = dist
        
        
#must have fully filled out matrix
l = len(target_names)
m = np.empty((l,l))
m[:] = np.NAN

for t1 in values.iterkeys():
    for t2 in values[t1].iterkeys():
        i = targets[t1]
        j = targets[t2]
        m[i][j] = values[t1][t2]
    
#check
for i in xrange(l):
    for j in xrange(l):
        if not np.isfinite(m[i][j]):
            print "Missing value for",targets[i],targets[j]

    
cPickle.dump((m, target_names), open('matrix.pickle','w'),-1)
