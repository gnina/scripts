#!/usr/bin/env python

import numpy as np
import matplotlib
from numpy import dtype
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, re, sklearn, collections, argparse, sys, os
import sklearn.metrics
import caffe
from caffe.proto.caffe_pb2 import NetParameter, SolverParameter
import google.protobuf.text_format as prototxt
import time


def write_model_file(model_file, template_file, train_file, test_file, root_folder, avg_rotations):
    param = NetParameter()
    with open(template_file, 'r') as f:
        prototxt.Merge(f.read(), param)
    for layer in param.layer:
        if layer.molgrid_data_param.source == 'TRAINFILE':
            layer.molgrid_data_param.source = train_file
        if layer.molgrid_data_param.source == 'TESTFILE':
            layer.molgrid_data_param.source = test_file
        if layer.molgrid_data_param.root_folder == 'DATA_ROOT':
            layer.molgrid_data_param.root_folder = root_folder
        if avg_rotations and 'TEST' in str(layer):
            layer.molgrid_data_param.rotate = 24 #TODO axial rotations aren't working
            #layer.molgrid_data_param.random_rotation = True
    with open(model_file, 'w') as f:
        f.write(str(param))


def write_solver_file(solver_file, train_model, test_models, type, base_lr, momentum, weight_decay,
                      lr_policy, gamma, power, random_seed, max_iter, snapshot_prefix):
    param = SolverParameter()
    param.train_net = train_model
    for test_model in test_models:
        param.test_net.append(test_model)
        param.test_iter.append(0) #don't test automatically
    param.test_interval = max_iter
    param.type = type
    param.base_lr = base_lr
    param.momentum = momentum
    param.weight_decay = weight_decay
    param.lr_policy = lr_policy
    param.gamma = gamma
    param.power = power
    param.display = 0 #don't print solver iterations
    param.random_seed = random_seed
    param.max_iter = max_iter
    param.snapshot_prefix = snapshot_prefix
    with open(solver_file,'w') as f:
        f.write(str(param))


def evaluate_test_net(test_net, n_tests, rotations):
    '''Evaluate a test network and return the results.
    The number of examples in the file the test_net reads from
    must equal n_tests, otherwise output will be misaligned.
    Can optionally take the average of multiple rotations of
    each example. Batch size should be 1 and other parameters
    should be set so that data access is sequential.'''

    #evaluate each example with each rotation
    y_true = []
    y_scores = [[] for _ in xrange(n_tests)]
    y_affinity = []
    y_predaffs = [[] for _ in xrange(n_tests)]
    losses = []
    for r in xrange(rotations):
        for x in xrange(n_tests): #TODO handle different batch sizes
            res = test_net.forward()
            if r == 0:
                y_true.append(float(res['labelout']))
            else:
                assert res['labelout'] == y_true[x] #sanity check
            y_scores[x].append(float(res['output'][0][1])) 
            if 'affout' in res:
                if r == 0:
                    y_affinity.append(float(res['affout']))
                else:
                    assert res['affout'] == y_affinity[x] #sanity check
                y_predaffs[x].append(float(res['predaff']))
            if 'loss' in res:
                losses.append(float(res['loss']))

    #average the scores from each rotation
    y_score = []
    y_predaff = []
    for x in xrange(n_tests):
        y_score.append(np.mean(y_scores[x]))
    if y_affinity:
        for x in range(n_tests):
            y_predaff.append(np.mean(y_predaffs[x]))

    #compute auc
    assert len(np.unique(y_true)) > 1
    auc = sklearn.metrics.roc_auc_score(y_true, y_score)

    #compute mean squared error (rmsd) of affinity (for actives only)
    if y_affinity:
        y_predaff = np.array(y_predaff)
        y_affinity = np.array(y_affinity)
        yt = np.array(y_true, np.bool)
        rmsd = sklearn.metrics.mean_squared_error(y_affinity[yt], y_predaff[yt])
    else:
        rmsd = None

    #compute mean loss
    if losses:
        loss = np.mean(losses)
    else:
        loss = None

    return auc, y_true, y_score, loss, rmsd, y_affinity, y_predaff


def count_lines(file):
    return sum(1 for line in open(file, 'r'))


'''Script for training a neural net model from gnina grid data.
A model template is provided along with training and test sets of the form
<prefix>[train|test][num].types
Test area, as measured by AUC, is periodically assessed.   At the end graphs are made.
Default is to do dynamic stepping of learning rate, but can explore other methods.
'''
def train_and_test_model(args, train_file, test_file, reduced_train_file, reduced_test_file, outname):
    '''run solver for iterations steps, on the given training file,
    every test_interval evaluate the roc of bothe the trainfile and the testfile
    return the full predictions for every tested iteration'''
    template = args.model
    test_interval = args.test_interval
    iterations = args.iterations
    
    if test_interval > iterations: #need to test once
        test_interval = iterations

    if args.avg_rotations:
        rotations = 24
    else:
        rotations = 1

    pid = os.getpid()

    test_model = 'traintest.%d.prototxt' % pid
    train_model = 'traintrain.%d.prototxt' % pid
    reduced_test_model = 'trainreducedtest.%d.prototxt' % pid
    reduced_train_model = 'trainreducedtrain.%d.prototxt' % pid
    write_model_file(test_model, template, train_file, test_file, args.data_root, args.avg_rotations)
    write_model_file(train_model, template, train_file, train_file, args.data_root, args.avg_rotations)
    test_models = [test_model, train_model]
    if args.reduced:
        write_model_file(reduced_test_model, template, train_file, reduced_test_file, args.data_root, args.avg_rotations)
        write_model_file(reduced_train_model, template, train_file, reduced_train_file, args.data_root, args.avg_rotations)
        test_models.extend([reduced_test_model, reduced_train_model])

    solverf = 'solver.%d.prototxt' % pid
    write_solver_file(solverf, test_model, test_models, args.solver, args.base_lr, args.momentum, args.weight_decay,
                      args.lr_policy, args.gamma, args.power, args.seed, iterations+args.cont, outname)
        
    if args.gpu >= 0:
        caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    
    solver = caffe.get_solver(solverf)
    if args.cont:
        solver.restore(solvername)
        solver.testall() #link testnets to train net

    if args.weights:
        solver.net.copy_from(args.weights)

    test_nets = {}
    test_nets['test'] = solver.test_nets[0], count_lines(test_file)
    test_nets['train'] = solver.test_nets[1], count_lines(train_file)
    if args.reduced:
        test_nets['reduced_test'] = solver.test_nets[2], count_lines(reduced_test_file)
        test_nets['reduced_train'] = solver.test_nets[3], count_lines(reduced_train_file)

    if args.cont:
        mode = 'a'    
        modelname = '%s_iter_%d.caffemodel' % (outname, args.cont)
        solvername = '%s_iter_%d.solverstate' % (outname, args.cont)
    else:
        mode = 'w'
    outfile = '%s.out' % outname
    out = open(outfile, mode, 0) #unbuffered

    #return evaluation results:
    #  auc, loss, and rmsd from each test
    #  y_true, y_score, y_aff, y_predaff from last test
    res = {}
    test_vals = {'auc':[], 'y_true':[], 'y_score':[], 'loss':[], 'rmsd':[], 'y_aff':[], 'y_predaff':[]}
    train_vals = {'auc':[], 'y_true':[], 'y_score':[], 'loss':[], 'rmsd':[], 'y_aff':[], 'y_predaff':[]}

    #also keep track of best test and train aucs
    best_test_auc = 0
    best_train_auc = 0
    best_train_interval = 0

    for i in xrange(iterations/test_interval):
        last_test = i == iterations/test_interval-1
        n_iter = args.cont + (i+1)*test_interval

        #train
        start = time.time()
        solver.step(test_interval)
        print "Iteration %d" % (args.cont + (i+1)*test_interval)
        print "Train time: %f" % (time.time()-start)

        #evaluate test set
        start = time.time()
        if args.reduced and not last_test:
            test_net, n_tests = test_nets['reduced_test']
        else:
            test_net, n_tests = test_nets['test']
        test_auc, y_true, y_score, _, test_rmsd, y_aff, y_predaff = evaluate_test_net(test_net, n_tests, rotations)
        print "Eval test time: %f" % (time.time()-start)

        if i > 0 and not (args.reduced and last_test): #check alignment
            assert np.all(y_true == test_vals['y_true'])
            assert np.all(y_aff == test_vals['y_aff'])

        test_vals['y_true'] = y_true
        test_vals['y_aff'] = y_aff
        test_vals['y_score'] = y_score
        test_vals['y_predaff'] = y_predaff
        print "Test AUC: %f" % test_auc
        test_vals['auc'].append(test_auc)
        if test_rmsd:
            print "Test RMSD: %f" % test_rmsd
            test_vals['rmsd'].append(test_rmsd)

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            if args.keep_best:
                solver.snapshot() #a bit too much - gigabytes of data

        #evaluate train set
        start = time.time()
        if args.reduced and not last_test:
            test_net, n_tests = test_nets['reduced_train']
        else:
            test_net, n_tests = test_nets['train']
        train_auc, y_true, y_score, train_loss, train_rmsd, y_aff, y_predaff = evaluate_test_net(test_net, n_tests, rotations)
        print "Eval train time: %f" % (time.time()-start)

        if i > 0 and not (args.reduced and last_test): #check alignment
            assert np.all(y_true == train_vals['y_true'])
            assert np.all(y_aff == train_vals['y_aff'])

        train_vals['y_true'] = y_true
        train_vals['y_aff'] = y_aff
        train_vals['y_score'] = y_score
        train_vals['y_predaff'] = y_predaff
        print "Train AUC: %f" % train_auc
        train_vals['auc'].append(train_auc)
        print "Train loss: %f" % train_loss
        train_vals['loss'].append(train_loss)
        if train_rmsd:
            print "Train RMSD: %f" % train_rmsd
            train_vals['rmsd'].append(train_rmsd)

        if train_auc > best_train_auc:
            best_train_auc = train_auc
            best_train_interval = i

        #check for improvement
        if args.dynamic:
            lr = solver.get_base_lr()
            if (i-best_train_interval) > args.step_when: #reduce learning rate
                lr *= args.step_reduce
                solver.set_base_lr(lr)
                best_train_interval = i #reset 
                best_train_auc = train_auc #the value too, so we can consider the recovery
            if lr < args.step_end:
                break #end early  

        #write out evaluation results
        out.write('%.4f %.4f %.6f %.6f' % (test_auc, train_auc, train_loss, solver.get_base_lr()))
        if None not in (test_rmsd, train_rmsd):
            out.write(' %.4f %.4f' % (test_rmsd, train_rmsd))
        out.write('\n')
        out.flush()

    out.close()
    solver.snapshot()
    del solver #free mem
    
    if not args.keep:
        os.remove(solverf)
        os.remove(test_model)
        os.remove(train_model)
        if args.reduced:
            os.remove(reduced_test_model)
            os.remove(reduced_train_model)

    return test_vals, train_vals


def write_finaltest_file(finaltest_file, y_true, y_score, footer, mode):

    with open(finaltest_file, mode) as out:
        for (label, score) in zip(y_true, y_score):
            out.write('%f %f\n' % (label, score))
        out.write(footer)


def last_iters_statistics(test_aucs, iterations, test_interval, last_iters):

    last_iters_test_aucs = []
    last_iters = 1000
    if last_iters > iterations:
        last_iters = iterations
    num_test_aucs = last_iters/test_interval
    for fold_test_aucs in test_aucs:
        a = fold_test_aucs[-num_test_aucs:]
        if a:
            last_iters_test_aucs.append(a)
    return np.mean(last_iters_test_aucs), np.max(last_iters_test_aucs), np.min(last_iters_test_aucs)


def training_plot(plot_file, train_series, test_series):

    fig = plt.figure()
    plt.plot(train_series, label='Train')
    plt.plot(test_series, label='Test')
    plt.legend(loc='best')
    plt.savefig(plot_file, bbox_inches='tight')


def plot_roc_curve(plot_file, fpr, tpr, auc):

    fig = plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, label='CNN (AUC=%.2f)' % auc, linewidth=4)
    plt.legend(loc='lower right',fontsize=20)
    plt.xlabel('False Positive Rate',fontsize=22)
    plt.ylabel('True Positive Rate',fontsize=22)
    plt.axes().set_aspect('equal')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.text(.05, -.25, txt, fontsize=22)
    plt.savefig(plot_file, bbox_inches='tight')


def plot_correlation(plot_file, y_aff, y_predaff, rmsd, r2):

    fig = plt.figure(figsize=(8,8))
    plt.plot(y_aff, y_predaff, 'o', label='RMSD=%.2f, R^2=%.3f (Pos)' % (rmsd, r2))
    plt.legend(loc='best', fontsize=20, numpoints=1)
    lo = np.min([np.min(y_aff), np.min(y_predaff)])
    hi = np.max([np.max(y_aff), np.max(y_predaff)])
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel('Experimental Affinity', fontsize=22)
    plt.ylabel('Predicted Affinity', fontsize=22)
    plt.axes().set_aspect('equal')
    plt.savefig(plot_file, bbox_inches='tight')        


def comma_separated_ints(ints):
     return [int(i) for i in ints.split(',') if i and i != 'None']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('-m','--model',type=str,required=True,help="Model template. Must use TRAINFILE and TESTFILE")
    parser.add_argument('-p','--prefix',type=str,required=True,help="Prefix for training/test files: <prefix>[train|test][num].types")
    parser.add_argument('-d','--data_root',type=str,required=False,help="Root folder for relative paths in train/test files",default='')
    parser.add_argument('-n','--foldnums',type=comma_separated_ints,required=False,help="Fold numbers to run, default is '0,1,2'",default='0,1,2')
    parser.add_argument('-a','--allfolds',action='store_true',required=False,help="Train and test file with all data folds, <prefix>.types",default=False)
    parser.add_argument('-i','--iterations',type=int,required=False,help="Number of iterations to run,default 10,000",default=10000)
    parser.add_argument('-s','--seed',type=int,help="Random seed, default 42",default=42)
    parser.add_argument('-t','--test_interval',type=int,help="How frequently to test (iterations), default 40",default=40)
    parser.add_argument('-o','--outprefix',type=str,help="Prefix for output files, default <model>.<pid>",default='')
    parser.add_argument('-g','--gpu',type=int,help='Specify GPU to run on',default=-1)
    parser.add_argument('-c','--cont',type=int,help='Continue a previous simulation from the provided iteration (snapshot must exist)',default=0)
    parser.add_argument('-k','--keep',action='store_true',default=False,help="Don't delete prototxt files")
    parser.add_argument('-r', '--reduced', action='store_true',default=False,help="Use a reduced file for model evaluation if exists(<prefix>[_reducedtrain|_reducedtest][num].types)")
    parser.add_argument('--avg_rotations', action='store_true',default=False, help="Use the average of the testfile's 24 rotations in its evaluation results")
    #parser.add_argument('-v,--verbose',action='store_true',default=False,help='Verbose output')
    parser.add_argument('--keep_best',action='store_true',default=False,help='Store snapshots everytime test AUC improves')
    parser.add_argument('--dynamic',action='store_true',default=False,help='Attempt to adjust the base_lr in response to training progress')
    parser.add_argument('--solver',type=str,help="Solver type. Default is SGD",default='SGD')
    parser.add_argument('--lr_policy',type=str,help="Learning policy to use. Default is inv.",default='inv')
    parser.add_argument('--step_reduce',type=float,help="Reduce the learning rate by this factor with dynamic stepping, default 0.5",default='0.5')
    parser.add_argument('--step_end',type=float,help='Terminate training if learning rate gets below this amount',default=0)
    parser.add_argument('--step_when',type=int,help="Perform a dynamic step (reduce base_lr) when training has not improved after this many test iterations, default 10",default=10)
    parser.add_argument('--base_lr',type=float,help='Initial learning rate, default 0.01',default=0.01)
    parser.add_argument('--momentum',type=float,help="Momentum parameters, default 0.9",default=0.9)
    parser.add_argument('--weight_decay',type=float,help="Weight decay, default 0.001",default=0.001)
    parser.add_argument('--gamma',type=float,help="Gamma, default 0.001",default=0.001)
    parser.add_argument('--power',type=float,help="Power, default 1",default=1)
    parser.add_argument('--weights',type=str,help="Set of weights to initialize the model with")
    parser.add_argument('-p2','--prefix2',type=str,required=True,help="Prefix for training/test files: <prefix>[train|test][num].types")
    parser.add_argument('-d2','--data_root2',type=str,required=False,help="Root folder for relative paths in train/test files",default='')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    
    #identify all train/test pairs
    pairs = []
    for i in args.foldnums:
        train = '%strain%d.types' % (args.prefix, i)
        test = '%stest%d.types' % (args.prefix, i)
        if not os.path.isfile(train):
            print 'error: %s does not exist' % train
            sys.exit(1)
        if not os.path.isfile(test):
            print 'error: %s does not exist' % test
            sys.exit(1)
        pairs.append((i, train, test))
    if args.allfolds:
        train = test = '%s.types' % args.prefix
        if not os.path.isfile(train):
            print 'error: %s does not exist' % train
            sys.exit(1)
        pairs.append(('all', train, test))
    
    if len(pairs) == 0:
        print "error: missing train/test files"
        sys.exit(1)
    
    for (i, train, test) in pairs:
        print train, test
    
    outprefix = args.outprefix
    if outprefix == '':
        outprefix = '%s.%d' % (os.path.splitext(os.path.basename(args.model))[0],os.getpid())
    
    mode = 'w'
    if args.cont:
        mode = 'a'
    
    test_aucs = []
    train_aucs = []
    test_rmsds = []
    train_rmsds = []
    all_y_true = []
    all_y_score = []
    all_y_aff = []
    all_y_predaff = []

    #train each pair
    for (i, trainfile, testfile) in pairs:
    
        outname = '%s.%s' % (outprefix, i)
        if args.allfolds and trainfile == testfile:
            crossval = False
            reducedtrainfile = reducedtestfile = trainfile.replace('.types', '_reduced.types')
        else:
            crossval = True
            reducedtrainfile = trainfile.replace('%strain' % args.prefix,'%s_reducedtrain' % args.prefix)
            reducedtestfile = trainfile.replace('%strain' % args.prefix,'%s_reducedtest' % args.prefix)

        if not os.path.isfile(reducedtrainfile):
            print 'error: %s does not exist' % reducedtrainfile
            sys.exit(1)
        if not os.path.isfile(reducedtestfile):
            print 'error: %s does not exist' % reducedtestfile
            sys.exit(1)

        test_vals, train_vals = train_and_test_model(args, trainfile, testfile, reducedtrainfile, reducedtestfile, outname)
        if not crossval:
            continue

        all_y_true.extend(test_vals['y_true'])
        all_y_score.extend(test_vals['y_score'])
        all_y_aff.extend(test_vals['y_aff'])
        all_y_predaff.extend(test_vals['y_predaff'])

        test_aucs.append(test_vals['auc'])
        train_aucs.append(train_vals['auc'])
        if test_vals['rmsd'] and train_vals['rmsd']:
            test_rmsds.append(test_vals['rmsd'])
            train_rmsds.append(train_vals['rmsd'])

        if np.mean(train_aucs) > 0:
            y_true, y_score, auc = test_vals['y_true'], test_vals['y_score'], test_vals['auc'][-1]
            write_finaltest_file('%s.finaltest' % outname, y_true, y_score, '# AUC %f\n' % auc, mode)

        if test_rmsds:
            y_aff, y_predaff, rmsd = test_vals['y_aff'], test_vals['y_predaff'], test_vals['rmsd'][-1]
            write_finaltest_file('%s.rmsd.finaltest' % outname, y_aff, y_predaff, '# RMSD %f\n' % rmsd, mode)

    #skip post processing if it's not a full crossvalidation
    if len(args.foldnums) <= 1:
        sys.exit(0)

    #average, min, max test AUC for last 1000 iterations
    last_iters = 1000
    avg_auc, max_auc, min_auc = last_iters_statistics(test_aucs, args.iterations, args.test_interval, last_iters)
    txt = 'For the last %s iterations:\nmean AUC=%.2f  max AUC=%.2f  min AUC=%.2f' % (last_iters, avg_auc, max_auc, min_auc)

    #due to early termination length of results may not be equivalent
    test_aucs = np.array(zip(*test_aucs))
    train_aucs = np.array(zip(*train_aucs))

    #average aucs across folds
    mean_test_aucs = test_aucs.mean(axis=1)
    mean_train_aucs = train_aucs.mean(axis=1)

    #write test and train aucs (mean and for each fold)
    with open('%s.test' % outprefix, mode) as out:
        for m, r in zip(mean_test_aucs, test_aucs):
            out.write('%s %s\n' % (m, ' '.join([str(x) for x in r])))

    with open('%s.train' % outprefix, mode) as out:
        for m, r in zip(mean_train_aucs, train_aucs):
            out.write('%s %s\n' % (m, ' '.join([str(x) for x in r])))    

    #training plot of mean auc across folds
    training_plot('%s_train.pdf' % outprefix, mean_train_aucs, mean_test_aucs)

    #roc curve for the last iteration - combine all tests
    if len(np.unique(all_y_true)) > 1:
        fpr, tpr, _ = sklearn.metrics.roc_curve(all_y_true, all_y_score)
        auc = sklearn.metrics.roc_auc_score(all_y_true, all_y_score)
        write_finaltest_file('%s.finaltest' % outprefix, all_y_true, all_y_score, '# AUC %f\n' % auc, mode)
        plot_roc_curve('%s_roc.pdf' % outprefix, fpr, tpr, auc)

    if test_rmsds:

        test_rmsds = np.array(zip(*test_rmsds))
        train_rmsds = np.array(zip(*train_rmsds))

        #average rmsds across folds
        mean_test_rmsds = test_rmsds.mean(axis=1)
        mean_train_rmsds = train_rmsds.mean(axis=1)

        #write test and train rmsds (mean and for each fold)
        with open('%s.rmsd.test' % outprefix, mode) as out:
            for m, r in zip(mean_test_rmsds, test_rmsds):
                out.write('%s %s\n' % (m, ' '.join([str(x) for x in r])))

        with open('%s.rmsd.train' % outprefix,mode) as out:
            for m, r in zip(mean_train_rmsds, train_rmsds):
                out.write('%s %s \n' % (m, ' '.join([str(x) for x in r])))

        #training plot of mean rmsd across folds
        training_plot('%s_rmsd_train.pdf' % outprefix, mean_train_rmsds, mean_test_rmsds)

        all_y_aff = np.array(all_y_aff)
        all_y_predaff = np.array(all_y_predaff)
        yt = np.array(all_y_true, dtype=np.bool)
        rmsdt = sklearn.metrics.mean_squared_error(all_y_aff[yt], all_y_predaff[yt])
        r2t = sklearn.metrics.r2_score(all_y_aff[yt], all_y_predaff[yt])
        write_finaltest_file('%s.rmsd.finaltest' % outprefix, all_y_aff, all_y_predaff, '# RMSD,R^2 %f %f\n' % (rmsdt, r2t), mode)

        plot_correlation('%s_rmsd.pdf' % outprefix, all_y_aff[yt], all_y_predaff[yt], rmsdt, r2t)

