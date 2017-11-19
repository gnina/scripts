#!/usr/bin/env python 

'''Take a prefix, run predictions, and generate evaluations for crystal, bestonly, 
and all test sets (take max affinity; if pose score is available also consider
max pose score).
Generates graphs and overall CV results.  Takes the prefix and (for now) assumes trial 0.
Will evaluate 100k model and best model prior to 100k, 50k and 25k
'''

import numpy as np
import os, sys
os.environ["GLOG_minloglevel"] = "10"
sys.path.append("/home/dkoes/git/gninascripts/")
sys.path.append("/net/pulsar/home/koes/dkoes/git/gninascripts/")

import train, predict
import matplotlib, caffe
import matplotlib.pyplot as plt
import glob, re, sklearn, collections, argparse, sys
import sklearn.metrics
import scipy.stats


def evaluate_fold(testfile, caffemodel):
    '''Evaluate the passed model and the specified test set.
    Assumes the .model file is named a certain way.
    Returns tuple:
    (correct, prediction, receptor, ligand, label (optional), posescore (optional))
    label and posescore are only provided is trained on pose data
    '''
    #figure out model name
    m = re.search(r'_(affinity.*)\.\d_iter_\d+.caffemodel',caffemodel)
    if not m:
        print "Couldn't parse:",caffemodel
    modelname = m.group(1)+".model"
    if not os.path.exists(modelname):
        m = re.search(r'_(affinity.*)_lr\S+\.\d_iter_\d+.caffemodel',caffemodel)
        if m:
           modelname = m.group(1)+".model"
        if not os.path.exists(modelname):
           m = re.search(r'_(affinity.*)_(Adam|SGD).*\.\d_iter_\d+.caffemodel',caffemodel)
           if m:
              modelname = m.group(1)+".model"
        if not os.path.exists(modelname): #try stripping off random seed
            m = re.search(r'(affinity.*)_\d+.model',modelname)
            if m:
                modelname = m.group(1)+".model"
	if not os.path.exists(modelname):
	    m = re.search(r'(affinity.*_relu)_.*\.model',modelname)
	    if m:
		modelname = m.group(1)+".model"
        if not os.path.exists(modelname):
           print modelname,"does not exist"
        
    caffe.set_mode_gpu()
    test_model = 'predict.%d.prototxt' % os.getpid()
    train.write_model_file(test_model, modelname, testfile, testfile, '')
    test_net = caffe.Net(test_model, caffemodel, caffe.TEST)
    lines = open(testfile).readlines()

    res = None
    i = 0 #index in batch
    correct = 0
    prediction = 0
    receptor = ''
    ligand = ''
    label = 0
    posescore = -1
    ret = []
    for line in lines:
        #check if we need a new batch of results
        if not res or i >= batch_size:
            res = test_net.forward()
            if 'output' in res:
                batch_size = res['output'].shape[0]
            else:
                batch_size = res['affout'].shape[0]
            i = 0

        if 'labelout' in res:
            label = float(res['labelout'][i])

        if 'output' in res:
            posescore = float(res['output'][i][1])

        if 'affout' in res:
            correct = float(res['affout'][i])

        if 'predaff' in res:
            prediction = float(res['predaff'][i])
            if not np.isfinite(prediction).all():
                os.remove(test_model)
                return [] #gracefully handle nan?

        #extract ligand/receptor for input file
        tokens = line.split()
        for t in xrange(len(tokens)):
            if tokens[t].endswith('gninatypes'):
                receptor = tokens[t]
                ligand = tokens[t+1]
                break
        
        #(correct, prediction, receptor, ligand, label (optional), posescore (optional))       
        if posescore < 0:
            ret.append((correct, prediction, receptor, ligand))
        else:
            ret.append((correct, prediction, receptor, ligand, label, posescore))
            
        i += 1 #batch index
        
    os.remove(test_model)
    return ret
    

def reduce_results(results, index):
    '''Return results with only one tuple for every receptor value,
    taking the one with the max value at index in the tuple (predicted affinity or pose score)
    '''
    res = dict() #indexed by receptor
    for r in results:
        name = r[2]
        if name not in res:
            res[name] = r
        elif res[name][index] < r[index]:
            res[name] = r
    return res.values()

def analyze_results(results, outname, uniquify=None):
    '''Compute error metrics from resuls.  RMSE, Pearson, Spearman.
    If uniquify is set, AUC and top-1 percentage are also computed,
    uniquify can be None, 'affinity', or 'pose' and is set with
    the all training set to select the pose used for scoring.
    Returns tuple:
    (RMSE, Pearson, Spearman, AUCpose, AUCaffinity, top-1)
    Writes (correct,prediction) pairs to outname.predictions
    '''

    #calc auc before reduction
    if uniquify and len(results[0]) > 5:
        labels = np.array([r[4] for r in results])
        posescores = np.array([r[5] for r in results])
        predictions = np.array([r[1] for r in results])
        aucpose = sklearn.metrics.roc_auc_score(labels, posescores)
        aucaff = sklearn.metrics.roc_auc_score(labels, predictions)

    if uniquify == 'affinity':
        results = reduce_results(results, 1)
    elif uniquify == 'pose':
        results = reduce_results(results, 5)

    predictions = np.array([r[1] for r in results])
    correctaff = np.array([abs(r[0]) for r in results])
    #(correct, prediction, receptor, ligand, label (optional), posescore (optional))    

    rmse = np.sqrt(sklearn.metrics.mean_squared_error(correctaff, predictions))
    R = scipy.stats.pearsonr(correctaff, predictions)[0]
    S = scipy.stats.spearmanr(correctaff, predictions)[0]
    out = open('%s.predictions'%outname,'w')
    for (c,p) in zip(correctaff,predictions):
        out.write('%f %f\n' % (c,p))
    out.write('#RMSD %f\n'%rmse)
    out.write('#R %f\n'%R)

    if uniquify and len(results[0]) > 5:
        labels = np.array([r[4] for r in results])
        top = np.count_nonzero(labels > 0)/float(len(labels))
        return (rmse, R, S, aucpose, aucaff, top)
    else:
        return (rmse, R, S)
    


name = sys.argv[1]

allresults = []
#for each test dataset
for testprefix in ['all','crystal','bestonly']:
    #find the relevant models for each fold
    testresults = {'best25': [], 'best50': [], 'best100': [], '100k': [], '250k': [], 'best250': [] }
    for fold in [0,1,2]:
        best25 = 0
        best50 = 0
        best100 = 0
        best250 = 0
        #identify best iteration models at each cut point for this fold
        for model in glob.glob('%s.%d_iter_*.caffemodel'%(name,fold)):
            m = re.search(r'_iter_(\d+).caffemodel', model)
            inum = int(m.group(1))
            if inum < 25000 and inum > best25:
                best25 = inum
            elif inum < 50000 and inum > best50:
                best50 = inum
            elif inum < 100000 and inum > best100:
                best100 = inum
            elif inum < 250000 and inum > best250:
                best250 = inum                
        #evalute this fold
        testfile = '../types/%s_0.5_0_test%d.types' % (testprefix,fold)
        if best25 > 0: testresults['best25'] += evaluate_fold(testfile, '%s.%d_iter_%d.caffemodel' % (name,fold,best25))
        if best50 > 0: testresults['best50'] += evaluate_fold(testfile, '%s.%d_iter_%d.caffemodel' % (name,fold,best50))
        if best100 > 0: testresults['best100'] += evaluate_fold(testfile, '%s.%d_iter_%d.caffemodel' % (name,fold,best100))
        if os.path.exists('%s.%d_iter_%d.caffemodel' % (name,fold,100000)): #100k
            testresults['100k'] += evaluate_fold(testfile, '%s.%d_iter_%d.caffemodel' % (name,fold,100000))
            last = '100k'
        if os.path.exists('%s.%d_iter_%d.caffemodel' % (name,fold,250000)): #possibility that the 100k model was output even if went to 250k
            if best250 > 0: testresults['best250'] += evaluate_fold(testfile, '%s.%d_iter_%d.caffemodel' % (name,fold,best250))
            testresults['250k'] += evaluate_fold(testfile, '%s.%d_iter_%d.caffemodel' % (name,fold,250000))
            last = '250k'
        
    for n in testresults.keys():
        if len(testresults[n]) != len(testresults[last]) or len(testresults[last]) == 0:
            print "Missing data with",n
            continue
        if testprefix == 'all':
            if len(testresults[n][0]) == 6:
                allresults.append( ('all_pose', n) + analyze_results(testresults[n],'all_pose_'+name+'_'+n,'pose'))
            allresults.append( ('all_affinity', n) + analyze_results(testresults[n],'all_affinity_'+name+'_'+n,'affinity'))
        else:    
            allresults.append( (testprefix, n) + analyze_results(testresults[n], testprefix+'_'+name+'_'+n) )
     
if len(sys.argv) > 2:
    out = open(sys.argv[2],'w')
else:
    out = sys.stdout

for a in allresults:
    out.write(' '.join(map(str,a))+'\n')
