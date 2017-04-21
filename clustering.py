#!/usr/bin/env python

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.Polypeptide import is_aa
from Bio import pairwise2
from multiprocessing import Pool
from functools import partial
import scipy.cluster.hierarchy
import numpy as np
import sys, argparse, bisect, re, os, fnmatch
import cPickle

''' Some amino acids have nonstandard residue names: 
http://ambermd.org/tutorials/advanced/tutorial1_adv/
HIE histidine H
HID histidine H
HEM hemoglobin? (http://www.bmsc.washington.edu/CrystaLinks/man/pdb/guide2.2_frame.html)
CYX cystenine C
CYM cystenine C'''


def getResidueString(structure):
    seq=''
    for model in structure:
        for residue in model.get_residues():
            if is_aa(residue.get_resname(), standard=True):
                seq+=(three_to_one(residue.get_resname()))
            else:
                resname = residue.get_resname()
            if resname == 'HIE' or resname == 'HID': seq+=('H')
            elif resname == 'CYX' or resname == 'CYM': seq+=('C')
            else: seq+=('X')
    return seq

def calcDistanceMatrix(targets, target_names):
    n = len(targets)
    pool = Pool()
    function = partial(cUTDM2, targets, target_names, n)
    mapOfTuples = pool.map(function, xrange(n))
    distanceMatrix = np.zeros((n,n))
    for tup in mapOfTuples:
        distanceMatrix[tup[0]][tup[1]] = distanceMatrix[tup[1]][tup[0]] = tup[3]
    return distanceMatrix

def cUTDM2(targets, target_names, n, r):
    for c in xrange(r+1,n,1):
        score = pairwise2.align.globalxx(targets[r], targets[c], score_only=True)
    length= max(len(targets[r]), len(targets[c]))
    distance = (length-score)/length
    print target_names[r],target_names[c], distance
    print r
    twoProteinsDistance = (r, c, distance)
    return twoProteinsDistance

def assignGroup(dists, t, group, explore):
    '''add any targets to group that are less than t away from what is in explore'''
    while explore:
      frontier = set()
      for i in explore:
        for j in xrange(len(dists)):
          if dists[i][j] < t and j not in group:
            group.add(j)
            frontier.add(j)
      explore.update(frontier)
      explore.discard(i)

def calcClusterGroups(dists, target_names, t):
    '''dists is a distance matrix (full) for target_names'''
    assigned = set()
    groups = []
    for i in xrange(len(dists)):
        if i not in assigned:
            group = set([i])
            assignGroup(dists, t, group, set([i]))
            groups.append(group)
            assigned.update(group)
            print i,len(group)
    ret = []
    for g in groups:
      group = [target_names[i] for i in g]
      ret.append(group)
    return ret

    
def createFolds(cluster_groups,cnum,args):
    #print cluster_groups
    sets = [[] for _ in xrange(cnum)]
    setlength=[0]*cnum
    target_numposes = [0]*len(cluster_groups)
    for i in xrange(len(cluster_groups)):
        for t in cluster_groups[i]:
            try:
                posenum = len(fnmatch.filter(os.listdir('%s%s'%(args.path,t)), '*.gninatypes'))
            except OSError: 
                print '%s gninatype files not found at %s%s'%(t,args.path,t)
                continue
        target_numposes[i] +=posenum
    for _ in xrange(len(cluster_groups)):
        maxindex =target_numposes.index( np.max(target_numposes))
        s =setlength.index(np.min(setlength))
        setlength[s] += target_numposes[maxindex]
        target_numposes[maxindex]= -1
        sets[s].extend(cluster_groups[maxindex])
    print 'groups created:'
    print setlength 
    return sets  
    
def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else: return -1

def crossvalidatefiles(sets,outname,cnum,args):
    print 'Making .types files'
    #create test/train files
    trainfiles = [open('%strain%d.types'%(outname,x),'w') for x in xrange(cnum)]
    testfiles = [open('%stest%d.types'%(outname,x),'w') for x in xrange(cnum)]
    input_file = open(args.input)
    for line in input_file:
        for word in re.findall(r'[\w]+', line):
            for i in xrange(cnum):
                if word in sets[i]:
                   testfiles[i].write(line)
                   for j in xrange(cnum):
                       if i != j:
                          trainfiles[j].write(line)
                   break
            else: #didn't find a matching work
                continue
            break

    input_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create train/test sets for cross-validation separating by sequence similarity of protein targets')
    parser.add_argument('-p','--pbdfiles',type=str,required=False,help="file with targetnames and paths to pbdfiles of targets and number of poses per target in input(separated by space)")
    parser.add_argument('-i','--input',type=str,required=False,help="Input .types file to create sets from")
    parser.add_argument('-o','--output',type=str,default='',help='Output file name,default=[test|train][input]')
    parser.add_argument('-n', '--number',type=int,default=3,help="number of folds to create. default=3")
    parser.add_argument('--threshold', type=float,default=.2,help='what percentage dissimilariy to cluster by. default: 80% similarity(.2 dissimilarity)')
    parser.add_argument('--path',type=str,default='/home/dkoes/PDBbind/general-set-with-refined/',help="path to gninatypes files")
    parser.add_argument('--cpickle',type=str,default='',help="cpickle file")
    parser.add_argument('-v','--verbose',action='store_true',default=False,help='verbose output')
    args = parser.parse_args()
    cnum = args.number

    outname = args.output
    #if outname =='' : outname = args.input.rsplit(".",1)[0]
    
    p= PDBParser(PERMISSIVE=1,QUIET=1)
    targets=[]
    target_names=[]

    if args.cpickle:
        (distanceMatrix, D, linkageMatrix, target_names) = cPickle.load(open(args.cpickle))
    else:
        file = open(args.pbdfiles)
        for line in file.readlines():
            data= line.split(" ")
            name = data[0]
            handle= data[1].strip()
            target_names.append(name)
            structure=p.get_structure(name,handle)
            seq=getResidueString(structure)
            targets.append(seq)
            file.close()
    print 'Number of targets: %d'%len(targets)

    if args.cpickle:
        cluster_groups = calcClusterGroups(distanceMatrix,target_names,args.threshold)
        if args.verbose:
            j=0
            for i in cluster_groups:
                j = j+1
                print j,':'
                for h in i:
                   print h
        folds = createFolds(cluster_groups,cnum, args)
    else:
        distanceMatrix = calcDistanceMatrix(targets, target_names)#distances are sequence dis-similarity so that a smaller distance corresponds to more similar sequence
        cluster_groups = calcClusterGroups(distanceMatrix,target_names,args.threshold)
        print '%d clusters created'%len(cluster_groups)
        if args.verbose:
            j=0
            for i in cluster_groups:
                j = j+1
                print j,':'
                for h in i:
                   print h
        folds = createFolds(cluster_groups,cnum, args)
    
    for f in xrange(len(folds)):
        print '\n%d targets in set %d'%(len(folds[f]), f)
        folds[f].sort()#output writing assumes folds are sorted
    
    crossvalidatefiles(folds,outname,cnum,args)
      
    
