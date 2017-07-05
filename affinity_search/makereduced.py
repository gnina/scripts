#!/usr/bin/env python

import sys,re, collections
'''reduce the provided example file to only have a single positive and single negative example per receptor (at most)'''

#first identify best rmsd
bestval = dict()
bestlig = dict()
for line in open(sys.argv[1]):
    vals = line.rstrip().split()
    if len(vals) < 6:
        continue
    rmsd = float(vals[5])
    rec = vals[2]
    if rec not in bestval or rmsd < bestval[rec]:
        bestval[rec] = rmsd
        bestlig[rec] = vals[3]
        
if len(bestlig) == 0:
    for line in open(sys.argv[1]):
        print line.rstrip()
else:
    diddecoy = set()
    for line in open(sys.argv[1]):
        vals = line.rstrip().split()
        rec = vals[2]
        if rec in bestlig and vals[3] == bestlig[rec]:
            print line.rstrip()
        elif int(vals[0]) == 0 and rec not in diddecoy:
            diddecoy.add(rec)
            print line.rstrip()
    
