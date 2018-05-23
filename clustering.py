#!/usr/bin/env python

from __future__ import print_function
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.Polypeptide import is_aa
from Bio import pairwise2
from multiprocessing import Pool, cpu_count
from functools import partial
import scipy.cluster.hierarchy
import numpy as np
import sys, argparse, bisect, re, os, fnmatch
import cPickle, collections
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity as fs
from rdkit.Chem.Fingerprints import FingerprintMols
import rdkit

''' Some amino acids have nonstandard residue names: 
http://ambermd.org/tutorials/advanced/tutorial1_adv/
HIE histidine H
HID histidine H
HEM hemoglobin? (http://www.bmsc.washington.edu/CrystaLinks/man/pdb/guide2.2_frame.html)
CYX cystenine C
CYM cystenine C'''


def getResidueStrings(structure):
    seqs = []
    for model in structure:
        for ch in model.get_chains():
            seq = ''
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
            seqs.append(seq)
    return seqs


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
    mindist = 1.0
    for seq1 in targets[a]:
        for seq2 in targets[b]:
            score = pairwise2.align.globalxx(seq1, seq2, score_only=True)
            length = max(len(seq1), len(seq2))
            distance = (length-score)/length
            if distance < mindist:
                mindist = distance
    #print (a,b,mindist)
    return (a, b, mindist)


def assignGroup(dists, ligandsim, t, t2, ligandt, explore, names):
    '''group targets that are less than t away from each other and what's in explore'''
    group = set(explore)
    while explore:
        frontier = set()
        for i in explore:
            for j in range(dists.shape[1]):
                if j not in group:
                    #add to the group if protein is close by threshold t (these are distances - default 0.5)
                    #also add if the ligands are more similar (not distance) than ligandt and 
                    #the protein is closer than t2 (default 0.8 - meaning more than 20% similar)
                    if dists[i][j] < t or (ligandsim[i][j] > ligandt and dists[i][j] < t2):
                        group.add(j)
                        frontier.add(j)                
                                        
        explore = frontier
    return group


def calcClusterGroups(dists, ligandsim, target_names, t, t2, ligandt):
    '''dists is a distance matrix (full) for target_names'''
    assigned = set()
    groups = []
    for i in range(dists.shape[0]):
        if i not in assigned:
            group = assignGroup(dists, ligandsim, t, t2, ligandt, set([i]),target_names)
            groups.append(group)
            assigned.update(group)
    return [set(target_names[i] for i in g) for g in groups]


def createFolds(cluster_groups, numfolds, target_lines, randomize):
    '''split target clusters into numfolds folds with balanced num poses per fold
       If randomize, will balance less well.
    '''
    folds = [[] for _ in range(numfolds)]
    fold_numposes = [0]*numfolds
    group_numposes = [0]*len(cluster_groups)
    foldmap = {}
    for i, group in enumerate(cluster_groups):
        #count num poses per group
        for target in group:            
            group_numposes[i] += len(target_lines[target])
    for _ in cluster_groups:
        #iteratively assign group with most poses to fold with fewest poses
        maxgroup = group_numposes.index(np.max(group_numposes))
        if randomize:
            space = np.max(fold_numposes) - np.array(fold_numposes)
            tot = np.sum(space)
            if tot == 0:
                minfold = np.random.choice(numfolds)
            else: #weighted selection, prefer spots with more free space
                choice = np.random.choice(tot)
                tot = 0
                for i in xrange(len(space)):
                    tot += space[i]
                    if choice < tot:
                        minfold = i
                        break
        else:
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


def crossvalidatefiles(folds, outname, numfolds, target_lines,reduce):
    #create test/train files
    trainfiles = [(open('{}train{}.types'.format(outname, i), 'w'),open('{}reducedtrain{}.types'.format(outname, i), 'w')) for i in range(numfolds)]
    testfiles = [(open('{}test{}.types'.format(outname, i), 'w'),open('{}reducedtest{}.types'.format(outname, i), 'w')) for i in range(numfolds)]
    target_set = set(sum(folds, []))
    
    for target in target_lines.iterkeys():
        for i in range(numfolds):
            if target in folds[i]:
                out = testfiles[i]
            else:
                out = trainfiles[i]
            for line in target_lines[target]:
                out[0].write(line)
                if np.random.random() < reduce:
                    out[1].write(line)
            


def loadFolds(inname, target_names, numfolds):
    #load test/train files
    trainfiles = [open('{}train{}.types'.format(inname,x),'r') for x in range(numfolds)]
    testfiles = [open('{}test{}.types'.format(inname,x),'r') for x in range(numfolds)]
    folds = [set() for _ in range(numfolds)]
    foldmap = {}
    target_set = set(target_names)
    for i in range(numfolds):
        for line in testfiles[i]:
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
        for line in trainfiles[i]:
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
                        print('warning: {} and {} are {:.3f}% similar but in different folds' \
                              .format(a_name, b_name, 100*(1-dists[a][b])))
                        ok = False
    if closest:
        print('{} and {} are the most similar targets in different folds ({:.3f}%)' \
              .format(closest[0], closest[1], 100*(1-min_dist)))
    return ok

def linesFromInput(infile):
    '''return dictionary mapping target name to all lines of types file
    Gets target name by looking for (\S+)/'''
    ret = collections.defaultdict(list)
    for line in open(infile):
        m = re.search('\s(\S+)/',line)
        if m:
            targ = m.group(1)
            ret[targ].append(line)
        else:
            print ("Could not identify target from line:\n%s"%line)
            sys.exit(1)
    return ret
        
        
def readPDBfiles(pdbfiles,ncpus=cpu_count()):
    pdb_parser = PDBParser(PERMISSIVE=1, QUIET=1)
    with open(pdbfiles, 'r') as file:
        pdblines = file.readlines()
    pool = Pool(ncpus)
    function = partial(loadTarget, pdb_parser)
    target_tups = pool.map(function, pdblines)
    target_names, targets = [], []
    for tup in sorted(target_tups):
        if not tup:
            continue
        else:
            target_names.append(tup[0])
            targets.append(tup[1])
    return target_names, targets


def loadTarget(pdb_parser, line):
    data = line.split(" ")
    target_name = data[0]
    target_pdb = data[1].strip()
    try:
        structure = pdb_parser.get_structure(target_name, target_pdb)
        seqs = getResidueStrings(structure)
        return (target_name, seqs)
    except IOError:
        print('warning: {} does not exist'.format(target_pdb))

def computeLigandSimilarity(target_names, fname):
    '''Read target (first col) and ligand (third col) from fname. 
    Return ligand similarity matrix indexed according to target_names'''
    fingerprints = dict()
    for line in open(fname):
        vals = line.split()
        targ = vals[0]
        ligfile = vals[2]
        smi = open(ligfile).readline().split()[0]
        mol = AllChem.MolFromSmiles(smi)
        if mol == None:
            mol = AllChem.MolFromSmiles(smi,sanitize=False)
        fp = FingerprintMols.FingerprintMol(mol)
        fingerprints[targ] = fp
    n = len(target_names)
    sims = np.zeros((n,n))
    for i in xrange(n):
        for j in xrange(i+1):
            fpi = fingerprints[target_names[i]]
            fpj = fingerprints[target_names[j]]
            sim = fs(fpi,fpj)
            sims[i,j] = sims[j,i] = sim
    return sims
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create train/test sets for cross-validation separating by sequence similarity of protein targets')
    parser.add_argument('--pdbfiles',type=str,help="file with target names and paths to pbdfiles of targets (separated by space)")
    parser.add_argument('--cpickle',type=str,help="cpickle file for precomputed distance matrix")
    parser.add_argument('-i','--input',type=str,help="input .types file to create folds from, it is assumed receptors in pdb named directories")
    parser.add_argument('-o','--output',type=str,default='',help='output name for clustered folds')
    parser.add_argument('-c','--check',type=str,help='input name for folds to check for similarity')
    parser.add_argument('-n', '--number',type=int,default=3,help="number of folds to create/check. default=3")
    parser.add_argument('-s','--similarity',type=float,default=0.5,help='what percentage similarity to cluster by. default=0.5')
    parser.add_argument('-s2','--similarity_with_similar_ligand',type=float,default=0.3,help='what percentage similarity to cluster by when ligands are similar default=0.3')
    parser.add_argument('-l','--ligand_similarity',type=float,default=0.9,help='similarity threshold for ligands, default=0.9')
    parser.add_argument('-d','--data_root',type=str,default='/home/dkoes/PDBbind/general-set-with-refined/',help="path to target dirs")
    parser.add_argument('--posedir',required=False,default='',help='subdir of target dirs where ligand poses are located')
    parser.add_argument('--randomize',required=False,type=int,default=None,help='randomize inputs to get a different split, number is random seed')
    parser.add_argument('-v','--verbose',action='store_true',default=False,help='verbose output')
    parser.add_argument('--reduce',type=float,default=0.05,help="Fraction to sample by for reduced files")
    args = parser.parse_args()

    threshold = 1 - args.similarity #similarity and distance are complementary
    threshold2 = 1 - args.similarity_with_similar_ligand
    ligand_threshold = args.ligand_similarity #this actually is a sim

    if args.cpickle:
        with open(args.cpickle, 'r') as file:
            (distanceMatrix, target_names,ligandsim) = cPickle.load(file)
    elif args.pdbfiles:
        if args.verbose: print("reading pdbs...")
        target_names, targets = readPDBfiles(args.pdbfiles)
        if args.verbose: print("calculating distance matrix...")
        distanceMatrix = calcDistanceMatrix(targets)
        ligandsim = computeLigandSimilarity(target_names, args.pdbfiles) #returns similarity matrix indexed according to target_names
        cPickle.dump((distanceMatrix, target_names, ligandsim), open(args.pdbfiles+'.pickle','w'),-1)
    else:
        exit('error: need --cpickle or --pdbfiles to compute target distance matrix')
    if args.verbose: print('Number of targets: {}'.format(len(target_names)))

        
    if args.randomize != None:
        np.random.seed(args.randomize)
        n = len(target_names)
        indices = np.random.choice(n,n,False)
        target_names = list(np.array(target_names)[indices])
        distanceMatrix = distanceMatrix[indices]
        distanceMatrix = distanceMatrix[:,indices]
        ligandsim = ligandsim[indices]
        ligandsim = ligandsim[:,indices]
        
    #de-link super common ligand
    
    #while True:
        #mi = np.argmax([np.sum(row == 1.0) for row in ligandsim[:,]])
        #cnt = np.sum(ligandsim[mi] == 1.0)
        #print(target_names[mi],cnt)
        #if cnt < 10:
            #break
        
        #for i in xrange(len(ligandsim)):
            #if ligandsim[mi][i] == 1.0:
                #ligandsim[ligandsim[:,i] == 1.0,i] = 0.0
    
    
    if args.input:
        target_lines = linesFromInput(args.input)
        cluster_groups = calcClusterGroups(distanceMatrix, ligandsim, target_names, threshold, threshold2, ligand_threshold)
        if args.verbose: print('{} clusters created'.format(len(cluster_groups)))
        if args.verbose:
            for i, g in enumerate(cluster_groups):
                print('Cluster {}: {}'.format(i, ' '.join(str(t) for t in g)))

        print("Max cluster size: %d" % np.max([len(c) for c in cluster_groups]))
        folds, foldmap = createFolds(cluster_groups, args.number, target_lines, args.randomize)
        for i, fold in enumerate(folds):
            print('{} targets in fold {}'.format(len(fold), i))

        if args.verbose: print('Making .types files')
        crossvalidatefiles(folds, args.output, args.number, target_lines, args.reduce)

    if args.check:
        folds, foldmap = loadFolds(args.check, target_names, args.number)
        print('Checking {} train/test folds for {:.3f}% similarity'.format(args.check, 100*args.similarity))
        checkFolds(distanceMatrix, target_names, threshold, foldmap)

