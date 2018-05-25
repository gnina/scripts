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

test_aucs, train_aucs = [], []
test_rmsds, train_rmsds = [], []
test_y_true, train_y_true = [], []
test_y_score, train_y_score = [], []
test_y_aff, train_y_aff = [], []
test_y_predaff, train_y_predaff = [], []
topresults = []

#train each pair
numfolds = 0
for i in train_test_files:

    outname = '%s.%s' % (outprefix, i)
    cont = 0
    checkname = '%s.CHECKPOINT'%outname
    if os.path.exists(checkname):
        (dontremove, prevsnap) = open(checkname).read().rstrip().split()[:2]
        m = re.search(r'%s_iter_(\d+)\.caffemodel'%outname,prevsnap)
        if m:
            cont = int(m.group(1))
            if cont >= trainargs.iterations:
                continue  #finished this fold
        else:
            print "Error parsing",checkname
            sys.exit(1)    
    
    results = train.train_and_test_model(trainargs, train_test_files[i], outname, cont)
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

    #aggregate results from different crossval folds
    if test.aucs:
        test_aucs.append(test.aucs)
        train_aucs.append(trainres.aucs)
        test_y_true.extend(test.y_true)
        test_y_score.extend(test.y_score)
        train_y_true.extend(trainres.y_true)
        train_y_score.extend(trainres.y_score)

    if test.rmsds:
        test_rmsds.append(test.rmsds)
        train_rmsds.append(trainres.rmsds)
        test_y_aff.extend(test.y_aff)
        test_y_predaff.extend(test.y_predaff)
        train_y_aff.extend(trainres.y_aff)
        train_y_predaff.extend(trainres.y_predaff)
        
    #run model to get calctop
    #first fine last model
    lastiter = 0
    cmodel = None
    for fname in glob.glob('*.%d_iter_*.caffemodel'%i):
        nums=(re.findall('\d+', fname ))
        new_iter=int(nums[-1])
        if new_iter>lastiter:
            lastiter=new_iter
            cmodel = fname
    topresults += calctop.evaluate_fold(train_test_files[i]['test'], cmodel, 'model.model',args.data_root)
    
#don't consider bad poses for affinity
test_y_aff = np.array(test_y_aff)
test_y_predaff = np.array(test_y_predaff)
exp = test_y_aff[test_y_aff > 0]
pred = test_y_predaff[test_y_aff > 0]
R = scipy.stats.pearsonr(exp, pred)[0]
rmse = np.sqrt(sklearn.metrics.mean_squared_error(exp,pred))
auc = sklearn.metrics.roc_auc_score(test_y_true, test_y_score)
top = calctop.find_top_ligand(topresults,1)/100.0

print d, R, rmse, auc, top

