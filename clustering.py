#!/usr/bin/env python

from __future__ import print_function
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
    seq = ''
    for model in structure:
        for residue in model.get_residues():
            resname = residue.get_resname()
            if is_aa(resname, standard=True):
                seq += three_to_one(resname)
            elif resname in {'HIE', 'HID'}:
                seq += 'H'
            elif resname in {'CYX', 'CYM'}:
                seq += 'C'
            else:
                seq += 'X'
    return seq


def calcDistanceMatrix(targets):
    '''compute full pairwise target distance matrix in parallel'''
    n = len(targets)
    pairs = [(r, c) for r in range(n) for c in range(r+1, n)] #upper triangle
    pool = Pool()
    function = partial(cUTDM2, targets)
    distanceTuples = pool.map(function, pairs)
    distanceMatrix = np.zeros((n, n))
    for (a, b, distance) in distanceTuples:
        distanceMatrix[a][b] = distanceMatrix[b][a] = distance
    return distanceMatrix


def cUTDM2(targets, pair):
    '''compute distance between target pair'''
    (a, b) = pair
    score = pairwise2.align.globalxx(targets[a], targets[b], score_only=True)
    length = max(len(targets[a]), len(targets[b]))
    distance = (length-score)/length
    return (a, b, distance)


def assignGroup(dists, t, explore):
    '''group targets that are less than t away from each other and what's in explore'''
    group = set(explore)
    while explore:
        frontier = set()
        for i in explore:
            for j in range(dists.shape[1]):
                if dists[i][j] < t and j not in group:
                    group.add(j)
                    frontier.add(j)
        explore = frontier
    return group


def calcClusterGroups(dists, target_names, t):
    '''dists is a distance matrix (full) for target_names'''
    assigned = set()
    groups = []
    for i in range(dists.shape[0]):
        if i not in assigned:
            group = assignGroup(dists, t, set([i]))
            groups.append(group)
            assigned.update(group)
    return [set(target_names[i] for i in g) for g in groups]


def createFolds(cluster_groups, numfolds, args):
    '''split target clusters into numfolds folds with balanced num poses per fold'''
    folds = [[] for _ in range(numfolds)]
    fold_numposes = [0]*numfolds
    group_numposes = [0]*len(cluster_groups)
    foldmap = {}
    for i, group in enumerate(cluster_groups):
        #count num poses per group
        for target in group:
            path = os.path.join(args.data_root, target, args.posedir)
            try:
                numposes = len(fnmatch.filter(os.listdir(path), '*.gninatypes'))
            except OSError: 
                print('warning: {} gninatype files not found at {}'.format(target, path))
                continue
            group_numposes[i] += numposes
    for _ in cluster_groups:
        #iteratively assign group with most poses to fold with fewest poses
        maxgroup = group_numposes.index(np.max(group_numposes))
        minfold = fold_numposes.index(np.min(fold_numposes))
        folds[minfold].extend(cluster_groups[maxgroup])
        fold_numposes[minfold] += group_numposes[maxgroup]
        group_numposes[maxgroup] = -1
        for t in cluster_groups[maxgroup]:
            foldmap[t] = minfold
    print('Poses per fold: {}'.format(fold_numposes))
    for f in folds:
        f.sort()
    return folds, foldmap


def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else: return -1


def crossvalidatefiles(folds, outname, numfolds, args):
    #create test/train files
    trainfiles = [open('{}train{}.types'.format(outname, i), 'w') for i in range(numfolds)]
    testfiles = [open('{}test{}.types'.format(outname, i), 'w') for i in range(numfolds)]
    target_set = set(sum(folds, []))
    with open(args.input, 'r') as file:
        lines = file.readlines()
    for line in lines:
        for word in re.findall(r'[\w]+', line):
            if word in target_set:
                target = word
                break
        for i in range(numfolds):
            if target in folds[i]:
                fold = i
                break
        for i in range(numfolds):
            if i == fold:
                testfiles[i].write(line)
            else:
                trainfiles[i].write(line)


def loadFolds(inname, target_names, numfolds):
    #load test/train files
    trainfiles = [open('{}train{}.types'.format(inname,x),'r') for x in range(numfolds)]
    testfiles = [open('{}test{}.types'.format(inname,x),'r') for x in range(numfolds)]
    folds = [set() for _ in range(numfolds)]
    foldmap = {}
    target_set = set(target_names)
    for i in range(numfolds):
        for line in testfiles[i].readlines():
            for word in re.findall(r'[\w]+', line):
                if word in target_set:
                    target = word
                    break
            for j in range(numfolds):
                if j == i:
                    folds[i].add(target)
                else:
                    assert target not in folds[j]
            foldmap[target] = i
        for line in trainfiles[i].readlines():
            for word in re.findall(r'[\w]+', line):
                if word in target_set:
                    target = word
                    break
            assert target not in folds[i]
    return folds, foldmap


def checkFolds(dists, target_names, threshold, foldmap):
    '''check that targets in different folds pass dissimilarity threshold'''
    ok = True
    n_targets = dists.shape[0]
    min_dist = np.inf
    closest = None
    for t in foldmap:
        if t not in set(target_names):
            print('warning: {} not found in distance matrix'.format(t))
    for a in range(n_targets):
        for b in range(a+1, n_targets):
            a_name = target_names[a]
            b_name = target_names[b]
            if a_name in foldmap and b_name in foldmap:
                if foldmap[a_name] != foldmap[b_name]:
                    if dists[a][b] < min_dist:
                        min_dist = dists[a][b]
                        closest = (a_name, b_name)
                    if dists[a][b] < threshold:
                        print('warning: {} and {} are similar ({:.5f}) but in different folds' \
                              .format(a_name, b_name, dists[a][b]))
                        ok = False
    if closest:
        print('{} and {} are the most similar targets in different folds ({:.5f})' \
              .format(closest[0], closest[1], min_dist))
    return ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create train/test sets for cross-validation separating by sequence similarity of protein targets')
    parser.add_argument('--pdbfiles',type=str,help="file with target names and paths to pbdfiles of targets (separated by space)")
    parser.add_argument('--cpickle',type=str,help="cpickle file for precomputed distance matrix")
    parser.add_argument('-i','--input',type=str,help="input .types file to create folds from")
    parser.add_argument('-o','--output',type=str,help='output name for clustered folds, default=[test|train][input]')
    parser.add_argument('-c','--check',type=str,help='input name for folds to check against dissimilarity threshold')
    parser.add_argument('-n', '--number',type=int,default=3,help="number of folds to create/check. default=3")
    parser.add_argument('-t','--threshold',type=float,default=.2,help='what percentage dissimilarity to cluster by. default: 80% similarity(.2 dissimilarity)')
    parser.add_argument('-d','--data_root',type=str,default='/home/dkoes/PDBbind/general-set-with-refined/',help="path to target dirs")
    parser.add_argument('--posedir',required=False,default='',help='subdir of target dirs where ligand poses are located')
    parser.add_argument('-v','--verbose',action='store_true',default=False,help='verbose output')
    args = parser.parse_args()

    targets = []
    target_names = []

    if args.cpickle:
        with open(args.cpickle, 'r') as file:
            (distanceMatrix, D, linkageMatrix, target_names) = cPickle.load(file)
    elif args.pdbfiles:
        p = PDBParser(PERMISSIVE=1, QUIET=1)
        with open(args.pdbfiles, 'r') as file:
            pdblines = file.readlines()
        for line in pdblines:
            data = line.split(" ")
            name = data[0]
            handle = data[1].strip()
            target_names.append(name)
            structure = p.get_structure(name, handle)
            seq = getResidueString(structure)
            targets.append(seq)
        #distances are sequence dis-similarity so that a smaller distance corresponds to more similar sequence
        distanceMatrix = calcDistanceMatrix(targets)
    else:
        exit('error: need --cpickle or --pdbfiles to compute target distance matrix')
    print('Number of targets: {}'.format(len(target_names)))

    if args.check:
        folds, foldmap = loadFolds(args.check, target_names, args.number)
        print('Checking {} for {} dissimilarity constraint'.format(args.check, args.threshold))
        checkFolds(distanceMatrix, target_names, args.threshold, foldmap)

    elif args.input and args.output:
        cluster_groups = calcClusterGroups(distanceMatrix, target_names, args.threshold)
        print('{} clusters created'.format(len(cluster_groups)))
        if args.verbose:
            for i, g in enumerate(cluster_groups):
                print('Cluster {}: {}'.format(i, ' '.join(str(t) for t in g)))

        folds, foldmap = createFolds(cluster_groups, args.number, args)
        for i, fold in enumerate(folds):
            print('{} targets in fold {}'.format(len(fold), i))

        print('Making .types files')
        crossvalidatefiles(folds, args.output, args.number, args)

