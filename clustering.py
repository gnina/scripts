#!/usr/bin/env python

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.Polypeptide import is_aa
from Bio import pairwise2
import scipy.cluster.hierarchy
import numpy as np
import sys, argparse, bisect, re, os, fnmatch

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

def calcUpperTriangleOfDistanceMatrix(targets):
    n = len(targets)
    distanceMatrix= []
    for r in xrange(n):
	for c in xrange(r+1,n,1):
	    score = pairwise2.align.globalxx(targets[r], targets[c], score_only=True)
	    length= max(len(targets[r]), len(targets[c]))
	    distance = (length-score)/length
	    distanceMatrix.append(distance)
    return distanceMatrix

def calcClusterGroups(distanceMatrix, target_names):
    distanceMatrix = np.array(distanceMatrix)
    linkageMatrix=scipy.cluster.hierarchy.average(distanceMatrix)
    cluster_indexs = scipy.cluster.hierarchy.fcluster(linkageMatrix, args.threshold)
    cluster_groups= [[] for _ in xrange(np.max(cluster_indexs))]
    for i in xrange(cluster_indexs.size):
	cluster_groups[cluster_indexs[i]-1].append(target_names[i])
    return cluster_groups
    
def createFolds(cluster_groups,cnum,args):
    #print cluster_groups
    if args.crossvalidate: #cnum should be 3
	sets = [[] for _ in xrange(cnum)]
	setlength=[0]*cnum
	target_numposes = [0]*len(cluster_groups)
	for i in xrange(len(cluster_groups)):
	    for t in cluster_groups[i]:
		try:
		    posenum = len(fnmatch.filter(os.listdir('%s/%s/gninatypes/'%(args.path,t)), '*.gninatypes'))
		except OSError: 
		    print '%s gninatype files not found at %s/%s/gninatypes/'%(t,args.path,t)
		    continue
		target_numposes[i] +=posenum
		#print '%s: %d '%(t,posenum)
	for _ in xrange(len(cluster_groups)):
	    maxindex =target_numposes.index( np.max(target_numposes))
	    s =setlength.index(np.min(setlength))
	    setlength[s] += target_numposes[maxindex]
	    target_numposes[maxindex]= -1
	    sets[s].extend(cluster_groups[maxindex])
	print 'groups created:'
	print setlength 
	return sets
	
    #if not crossvalidate
    folds = [[] for _ in xrange(cnum)]
    if cnum == 1:
	for cluster in cluster_groups: folds[0].extend(cluster)
	return folds
    index_list = np.arange(0, len(cluster_groups))
    n= len(index_list)    
    if args.repeat:
	for i in xrange(0,cnum,2):
	    np.random.shuffle(index_list)
	    for j in xrange(n):
		if j< (n/2): folds[i].extend(cluster_groups[index_list[j]])
		else: folds[i+1].extend(cluster_groups[index_list[j]])	    
    else:
	np.random.shuffle(index_list)
	for i in xrange(n):
	    folds[i%cnum].extend(cluster_groups[index_list[i]])
    return folds
def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else: return -1

def crossvalidatefiles(sets,outname,cnum,args):
    print 'Making .types files'
    #create test/train files
    sets_index=0
    for i in xrange(cnum):#should be 3
	#train portion
	input_file = open(args.input, 'r')
	filename='%strain%d.types'%(outname, i)
	out_file = open(filename, 'w')
	next_i = sets_index +1
	if next_i > len(sets)-1: next_i = 0
	for line in input_file:
	    line_words = re.findall(r"[\w]+", line)
	    for (t,t2) in zip(sets[sets_index], sets[next_i]):
		if index(line_words,t) != -1: 
		    out_file.write(line)
		    break
		elif index(line_words,t2) != -1: 
		    out_file.write(line)
		    break
	input_file.close()
	out_file.close()
	next_i += 1
	if next_i > len(sets)-1: next_i = 0
	print filename
	#test portion
	filename = '%stest%d.types'%(outname, i)
	out_file = open(filename, 'w')
	input_file = open(args.input, 'r')
	for line in input_file:
	    line_words = re.findall(r"[\w]+", line)
	    for t in sets[next_i]:
		if index(line_words,t) != -1: 
		    out_file.write(line)
		    break
	input_file.close()
	out_file.close()
	sets_index+=1
	print filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create train/test sets for cross-validation separating by sequence similarity of protein targets')
    parser.add_argument('-p','--pbdfiles',type=str,required=True,help="file with targetnames and paths to pbdfiles of targets and number of poses per target in input(separated by space)")
    parser.add_argument('-i','--input',type=str,required=False,help="Input .types file to create sets from")
    parser.add_argument('-o','--output',type=str,default='',help='Output file name,default=[test|train][input]')
    parser.add_argument('-n', '--number',type=int,default=3,help="number of folds to create. default=3")
    parser.add_argument('--crossvalidate', action='store_false', default=True,help='create test and train sets')
    parser.add_argument('-t', '--test_train', action='store_false', default=True,help='create test and train sets')
    parser.add_argument('-r', '--repeat', action='store_false', default=True,help='if true poses can repeat between folds')
    parser.add_argument('--threshold', type=float,default=.2,help='what percentage dissimilariy to cluster by. default: 80% similarity(.2 dissimilarity)')
    parser.add_argument('--path',type=str,default='/home/dkoes/DUDe',help="path to gninatypes files")
    args = parser.parse_args()
    if args.crossvalidate:
	args.test_train = True
	cnum=3
    else: cnum = args.number*2
    if not args.test_train: args.repeat = False
    outname = args.output
    if outname =='' : outname = args.input.rsplit(".",1)[0]
    
    p= PDBParser(PERMISSIVE=1,QUIET=1)
    targets=[]
    target_names=[]

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
    if not args.test_train and args.number == 1 and not args.crossvalidate: 
	folds = []
	folds.append(target_names)
    else:
	distanceMatrix = calcUpperTriangleOfDistanceMatrix(targets)#distances are sequence dis-similarity so that a smaller distance corresponds to more similar sequence
	cluster_groups = calcClusterGroups(distanceMatrix,target_names)
	print '%d clusters created'%len(cluster_groups)
	folds = createFolds(cluster_groups,cnum, args)
    
    for f in xrange(len(folds)):
	print '\n%d targets in set %d'%(len(folds[f]), f)
	folds[f].sort()#output writing assumes folds are sorted
	
    
    if args.crossvalidate: crossvalidatefiles(folds,outname,cnum,args)
    else: 
	if args.input == '':
	    fold_index=0
	    for i in xrange(args.number):
		if args.test_train:
		    out_file = open('%stest%d'%(outname, i), 'w')
		    for line in folds[fold_index]:
			out_file.write(line)
		    out_file.close()
		    fold_index +=1
		    print '%stest%d.types'%(outname, i)
		    out_file = open('%strain%d'%(outname, i), 'w')
		    for line in folds[fold_index]:
			out_file.write(line)
		    out_file.close()
		    fold_index+=1
		    print '%strain%d.types'%(outname, i)
		else:
		    out_file = open('%s%d'%(outname, i), 'w')
		    for line in folds[fold_index]:
			out_file.write(line)
		    out_file.close()
		    fold_index +=1
		    print '%s%d.types'%(outname, i)
	else: #create .types files using input
	    fold_index = 0
	    if args.test_train:
		for i in xrange(args.number):
		    input_file = open(args.input, 'r')
		    out_file = open('%stest%d.types'%(outname, i), 'w')
		    for line in input_file:
			line_words = re.findall(r"[\w]+", line)                  
			for t in folds[fold_index]:
			    if index(line_words,t) != -1: 
				out_file.write(line)
				break
		    out_file.close()
		    input_file.close()
		    fold_index+=1
		    print '%stest%d.types'%(outname, i)
		    input_file = open(args.input, 'r')
		    out_file = open('%strain%d.types'%(outname, i), 'w')
		    for line in input_file:
			line_words = re.findall(r"[\w]+", line)
			for t in folds[fold_index]:
			    if index(line_words,t) != -1: 
				out_file.write(line)
				break
		    out_file.close()
		    input_file.close()
		    fold_index+=1
		    print '%strain%d.types'%(outname, i)
	    else:
		input_file = open(args.input, 'r')
		out_file = open('%s%d.types'%(outname, i), 'w')
		for line in input_file:
		    for t in folds[fold_index]:
			if index(re.findall(r"[\w]+",line),t) != -1: 
			    out_file.write(line)
			    break
		out_file.close()
		input_file.close()
		fold_index+=1
		print '%s%d.types'%(outname, i)
	if args.test_train: print '%d files generated'%(args.number*2)
	else: print '%d files generated'%args.number
	       
    
