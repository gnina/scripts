#!/usr/bin/env python 

'''Take a prefix, model name and output and generate predictions for
a cross-docked formatted dataset.  Will use the last model

'''

import numpy as np
import os, sys
#os.environ["GLOG_minloglevel"] = "10"
sys.path.append("/home/dkoes/git/gninascripts/")
sys.path.append("/net/pulsar/home/koes/dkoes/git/gninascripts/")
sys.path.append("/home/dkoes/git/gninascripts/affinity_search")
sys.path.append("/net/pulsar/home/koes/dkoes/git/gninascripts/affinity_search")
import train, predict
import matplotlib, caffe
import matplotlib.pyplot as plt
import glob, re, sklearn, collections, argparse, sys
import sklearn.metrics
import scipy.stats

from evaluate import evaluate_fold


def reduce_results(results, index, which):
    '''Return results with only one tuple for every pocket-ligand value,
    taking the one with the max value at index in the tuple (predicted affinity or pose score)
    '''
    res = dict() #indexed by pocketligand
    for r in results:
        lname = r[3]
        m = re.search(r'(\S+)/????_?_rec_????_(\S+)_lig',lname)
        pocket = m.group(1)
        lig = m.group(2)
        key = pocket+':'+lig
        if key not in res:
            res[key] = r
        else:
            if which == 'small': #select smallest by index
                if res[key][index] > r[index]:
                    res[key] = r
            elif res[key][index] < r[index]:
                res[key] = r
                    
    return res.values()

def analyze_cross_results(results,outname,uniquify):
    '''Compute error metrics from resulst.  
    results is formated: (correct, prediction, receptor, ligand, label, posescore,rmsd)
    This is assumed to be a cross docked input, where receptor filename is
    POCKET/PDB_CH_rec_0.gninatypes
    and the ligand is
    POCKET/PDB1_CH_rec_PDB2_lig_...gninatypes
    
    
    RMSE, Pearson, Spearman, AUC and top-1 percentage are computed.
    
    select can be pose or rmsd.  With pose we use the best pose scoring pose
    of all ligand poses in a pocket.  For rmsd we use the lowest rmsd line for
    that ligand in the pocket.
    
    AUC is calculated before any reduction.
    
    Writes the reduced set to outname.predictions
    '''

    #calc auc before reduction
    labels = np.array([r[4] for r in results])
    posescores = np.array([r[5] for r in results])
    predictions = np.array([r[1] for r in results])
    aucpose = sklearn.metrics.roc_auc_score(labels, posescores)
    aucaff = sklearn.metrics.roc_auc_score(labels, predictions)

    if uniquify == 'rmsd':
        results = reduce_results(results, 6, 'small')
    elif uniquify == 'affinity':
        results = reduce_results(results, 1, 'large')
    elif uniquify == 'pose':
        results = reduce_results(results, 5, 'large')

    predictions = np.array([r[1] for r in results])
    correctaff = np.array([abs(r[0]) for r in results])
    #(correct, prediction, receptor, ligand, label (optional), posescore (optional))    

    rmse = np.sqrt(sklearn.metrics.mean_squared_error(correctaff, predictions))
    R = scipy.stats.pearsonr(correctaff, predictions)[0]
    S = scipy.stats.spearmanr(correctaff, predictions)[0]
    out = open('%s.predictions'%outname,'w')
    out.write('aff,pred,rec,lig,lab,score,rmsd\n')
    for res in results:
        out.write(','.join(res)+'\n')
    out.write('#RMSD %f\n'%rmse)
    out.write('#R %f\n'%R)

    labels = np.array([r[4] for r in results])
    top = np.count_nonzero(labels > 0)/float(len(labels))
    return (rmse, R, S, aucpose, aucaff, top)

    

if __name__ == '__main__':
    if len(sys.argv) <= 4:
        print "Need data root, caffemodel prefix,  modelname, output name and test prefix(es)"
        sys.exit(1)
        
    datadir = sys.argv[1]
    name = sys.argv[2]
    modelname = sys.argv[3]
    out = open(sys.argv[4],'w')

    allresults = []
    last = None
    #for each test dataset
    for testprefix in sys.argv[5:]:
        print testprefix        
        #find the relevant models for each fold
        
        testresuls = []
        for fold in [0,1,2]: #blah! hard coded
            lastm = 0
            #identify last iteration model for this fold
            for model in glob.glob('%s.%d_iter_*.caffemodel'%(name,fold)):
                m = re.search(r'_iter_(\d+).caffemodel', model)
                inum = int(m.group(1))
                if inum > lastm:
                    lastm = inum
                                 
            #evalute this fold
            testfile = '%stest%d.types' % (testprefix,fold)            
            testresults += evaluate_fold(testfile, '%s.%d_iter_%d.caffemodel' % (name,fold,lastm), modelname, datadir, True)
            
        if len(testresults) == 0:
            print "Missing data with",testprefix
        assert(len(testresults[0]) == 6)
        
        allresults.append( (testname,'pose') + analyze_cross_results(testresults,testname+'_pose','pose')
        allresults.append( (testname,'rmsd') + analyze_cross_results(testresults,testname+'_rmsd','rmsd')
        allresults.append( (testname,'pose') + analyze_cross_results(testresults,testname+'_affinity','affinity')

         
    for a in allresults:
        out.write(' '.join(map(str,a))+'\n')
