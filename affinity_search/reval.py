#!/usr/bin/env python

'''Reevaluates *.caffemodel models in specified directory.  Return same result as runline. Assume directory was created with do1request/runline.'''

import sys,os
def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))
    
sys.path.append(get_script_path()+'/..') #train
import re,argparse, tempfile, os,glob
import makemodel
import socket
import train
import numpy as np
import sklearn.metrics
import scipy.stats
import calctop, predict

class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


parser = argparse.ArgumentParser(description='Evaluate models in directory and report results.')
parser.add_argument('--data_root',type=str,help='Location of gninatypes directory',default='')
parser.add_argument('--prefix',type=str,help='Prefix, not including split',default='../data/refined/all_0.5_')
parser.add_argument('--split',type=int,help='Which predefined split to use',required=True)
parser.add_argument('--dir',type=str,help='Directory to use',required=True)
args = parser.parse_args()


os.chdir(args.dir)
prefix = '%s%d_'% (args.prefix,args.split)

#collect only the latest caffemodel file for each fold
models = {}
for caffefile in glob.glob('*.caffemodel'):
    m = re.search(r'\S+(\d+)_iter_(\d+).caffemodel',caffefile)
    if m:
        fold = int(m.group(1))
        iter = int(m.group(2))
        if fold not in models or models[fold][1] < iter:
            models[fold] = (caffefile,iter)
        

#for each fold, collect the predictions
predictions = [] #a list of strings
topresults = []
for fold in models:
    (caffefile, iter) = models[fold]
    testfile = prefix+'test%d.types'%fold
    pargs = predict.parse_args(['-m','model.model','-w',caffefile,'-d',args.data_root,'-i',testfile])
    predictions += predict.predict(pargs)
    topresults += calctop.evaluate_fold(testfile,caffefile,'model.model',args.data_root)


#parse prediction lines 
expaffs = []
predaffs = []
scores = []
labels = []
for line in predictions:
    if line.startswith('#'):
        continue
    vals = line.split()
    score = float(vals[0])
    predaff = float(vals[1])
    label = float(vals[2])
    aff = float(vals[3])
    scores.append(score)
    labels.append(label)
    if aff > 0:
        expaffs.append(aff)
        predaffs.append(predaff)
    
#don't consider bad poses for affinity
R = scipy.stats.pearsonr(expaffs, predaffs)[0]
rmse = np.sqrt(sklearn.metrics.mean_squared_error(expaffs,predaffs))
auc = sklearn.metrics.roc_auc_score(labels, scores)
top = calctop.find_top_ligand(topresults,1)/100.0

print args.dir, R, rmse, auc, top

