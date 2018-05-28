#!/usr/bin/env python

'''Read a line formated like a spearmint results.dat line,
construct the corresponding model, run the model with cross validation,
and print the results; dies with error if parameters are invalid'''

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
import calctop
from evaluate import evaluate_fold, analyze_results
from train import Namespace

class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


parser = argparse.ArgumentParser(description='Run single model line and report results.')
parser.add_argument('--line',type=str,help='Complete line',required=True)
parser.add_argument('--seed',type=int,help='Random seed',default=0)
parser.add_argument('--split',type=int,help='Which predefined split to use',default=0)
parser.add_argument('--data_root',type=str,help='Location of gninatypes directory',default='')
parser.add_argument('--prefix',type=str,help='Prefix, not including split',default='../data/refined/all_0.5_')
parser.add_argument('--dir',type=str,help='Directory to use')
args = parser.parse_args()

linevals = args.line.split()[2:]

opts = makemodel.getoptions()

if len(linevals) != len(opts):
    print "Wrong number of options in line (%d) compared to options (%d)" %(len(linevals),len(opts))

params = dict()
for (i,(name,vals)) in enumerate(sorted(opts.items())):
    v = linevals[i]
    if v == 'False':
        v = 0
    if v == 'True':
        v = 1
    if type(vals) == tuple:
        if type(vals[0]) == int:
            v = int(v)
        elif type(vals[0]) == float:
            v = float(v)
    elif isinstance(vals, makemodel.Range):
        v = float(v)
    params[name] = v


params = Bunch(params)

model = makemodel.create_model(params)

host = socket.gethostname() 

if args.dir:
    d = args.dir
    try:
        os.makedirs(d)
    except OSError:
        pass
else:
    d = tempfile.mkdtemp(prefix=host+'-',dir='.')

os.chdir(d)
mfile = open('model.model','w')
mfile.write(model)
mfile.close()

#get hyperparamters
base_lr = 10**params.base_lr_exp
momentum=params.momentum
weight_decay = 10**params.weight_decay_exp
solver = params.solver

#setup training
prefix = '%s%d_'% (args.prefix,args.split)
trainargs = train.parse_args(['--seed',str(args.seed),'--prefix',prefix,'--data_root',
    args.data_root,'-t','1000','-i','250000','-m','model.model','--checkpoint',
    '--reduced','-o',d,'--momentum',str(momentum),'--weight_decay',str(weight_decay),
    '--base_lr',str(base_lr),'--solver',solver,'--dynamic','--lr_policy','fixed'])[0]

train_test_files = train.get_train_test_files(prefix=prefix, foldnums=None, allfolds=False, reduced=True, prefix2=None)
if len(train_test_files) == 0:
    print "error: missing train/test files",prefix
    sys.exit(1)


outprefix = d
#train 
numfolds = 0
for i in train_test_files:

    outname = '%s.%s' % (outprefix, i)    
    results = train.train_and_test_model(trainargs, train_test_files[i], outname)
    test, trainres = results

    if not np.isfinite(np.sum(trainres.y_score)):
        print "Non-finite trainres score"
        sys.exit(-1)
    if not np.isfinite(np.sum(test.y_score)):
        print "Non-finite test score"
        sys.exit(-1)
    if not np.isfinite(np.sum(trainres.y_predaff)):
        print "Non-finite trainres aff"
        sys.exit(-1)
    if not np.isfinite(np.sum(test.y_predaff)):
        print "Non-finite test aff"
        sys.exit(-1)                        

#once all folds are trained, test and evaluate them
testresults = []
for i in train_test_files:
    
    #get latest model file for this fold
    lasti = -1
    caffemodel = ''
    for model in glob.glob('%s.%d_iter_*.caffemodel'%(outprefix,i)):
        m = re.search(r'_iter_(\d+).caffemodel', model)
        inum = int(m.group(1))
        if inum > lasti:
            lasti = inum
            caffemodel = model
    if lasti == -1:
        print "Couldn't find valid caffemodel file %s.%d_iter_*.caffemodel"%(outprefix,i)
        sys.exit(-1)
        
    testresults += evaluate_fold(train_test_files[i]['test'], caffemodel, 'model.model',trainargs.data_root)
    

(rmse, R, S, aucpose, aucaff, top) = analyze_results(testresults,'%s.summary'%outprefix,'pose')

print d, R, rmse, aucpose, top

