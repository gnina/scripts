#!/usr/bin/env python

import sys,re, collections
'''convert the provided example file to only have a single positive affinity
for the best rmsd example'''

#first identify best rmsd
bestval = dict()
bestlig = dict()
for line in open(sys.argv[1]):
#    0 1a30/1a30_rec.gninatypes 1a30/1a30_ligand_0.gninatypes # 8.46937 -8.3175
    vals = line.rstrip().split()
    rmsd = float(vals[5])
    rec = vals[2]
    if rec not in bestval or rmsd < bestval[rec]:
        bestval[rec] = rmsd
        bestlig[rec] = vals[3]
        
for line in open(sys.argv[1]):
    vals = line.rstrip().split()
    rec = vals[2]
    if vals[3] == bestlig[rec] or float(vals[1]) < 0:
        print line.rstrip()
    else:
        print vals[0],-float(vals[1]),' '.join(vals[2:])
    
