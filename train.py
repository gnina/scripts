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

    return (auc, y_true, y_score, loss, rmsd, y_affinity, y_predaff)


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
        write_model_file(reduced_test_model, template, train_file, reduced_test_file, args.avg_rotations)
        write_model_file(reduced_train_model, template, train_file, reduced_train_file, args.avg_rotations)
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
    test_nets['test'] = solver.test_nets[0]
    test_nets['train'] = solver.test_nets[1]
    if args.reduced:
        test_nets['reduced_test'] = solver.test_nets[2]
        test_nets['reduced_train'] = solver.test_nets[3]

    n_test_lines = sum(1 for line in open(test_file))
    n_train_lines = sum(1 for line in open(train_file))
    if args.reduced:
        n_reduced_test_lines = sum(1 for line in open(reduced_test_file))
        n_reduced_train_lines = sum(1 for line in open(redcued_train_file))

    if args.cont:
        mode = 'a'    
        modelname = '%s_iter_%d.caffemodel' % (outname, args.cont)
        solvername = '%s_iter_%d.solverstate' % (outname, args.cont)
    else:
        mode = 'w'
    outfile = '%s.out' % outname
    out = open(outfile, mode, 0) #unbuffered

    #return all evaluation results from each test interval
    #(auc, y_true, y_pred, loss, rmsd, y_affinity, y_predaff)
    test_vals = []
    train_vals = []

    #also keep track of best test and train aucs
    best_test_auc = 0
    best_train_auc = 0
    best_train_interval = 0

    for i in xrange(iterations/test_interval):
        last_test = i == iterations/test_interval-1
        n_iter = args.cont + i*test_interval

        #train
        start = time.time()
        solver.step(test_interval)
        print "Iteration %d" % (args.cont + (i+1)*test_interval)
        print "Train time: %f" % (time.time()-start)

        #evaluate test set
        start = time.time()
        if args.reduced and not last_test:
            test_net = test_nets['reduced_test']
            n_tests = n_reduced_test_lines
        else:
            test_net = test_nets['test']
            n_tests = n_test_lines
        test_vals.append(evaluate_test_net(test_net, n_tests, rotations))
        print "Eval test time: %f" % (time.time()-start)

        test_auc = test_vals[-1][0]
        test_rmsd = test_vals[-1][4]
        print "Test AUC: %f\nTest RMSD: %f" % (test_auc, test_rmsd)
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            if args.keep_best:
                solver.snapshot() #a bit too much - gigabytes of data

        #evaluate train set
        start = time.time()
        if args.reduced and not last_test:
            test_net = test_nets['reduced_train']
            n_tests = n_reduced_train_lines
        else:
            test_net = test_nets['train']
            n_tests = n_train_lines
        train_vals.append(evaluate_test_net(test_net, n_tests, rotations))
        print "Eval train time: %f" % (time.time()-start)

        train_auc = train_vals[-1][0]
        train_loss = train_vals[-1][3]
        train_rmsd = train_vals[-1][4]
        print "Train AUC: %f\nTrain loss: %f\nTrain RMSD: %f" % (train_auc, train_loss, train_rmsd)
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
        pairs.append((train, test))
    if args.allfolds:
        train = test = '%s.types' % args.prefix
        if not os.path.isfile(train):
            print 'error: %s does not exist' % train
            sys.exit(1)
        pairs.append((train, test))
    
    if len(pairs) == 0:
        print "error: missing train/test files"
        sys.exit(1)
    
    for (train, test) in pairs:
        print train, test
    
    outprefix = args.outprefix
    if outprefix == '':
        outprefix = '%s.%d' % (os.path.splitext(os.path.basename(args.model))[0],os.getpid())
    
    mode = 'w'
    if args.cont:
        mode = 'a'
    
    #train each pair
    testaucs = []
    trainaucs = []
    testrmsds = []
    trainrmsds = []
    alltest = []
    for (train,test) in pairs:
    
        if args.allfolds and train == test:
            crossval = False
            outname = '%s.all' % outprefix
            reducedtrainfile = reducedtestfile = train.replace('.types', '_reduced.types')
        else:
            crossval = True
            m = re.search('%strain(\d+)'%args.prefix,train)
            outname = '%s.%s' % (outprefix,m.group(1))
            reducedtrainfile = train.replace('%strain' % args.prefix,'%s_reducedtrain' % args.prefix)
            reducedtestfile = train.replace('%strain' % args.prefix,'%s_reducedtest' % args.prefix)

        if not os.path.isfile(reducedtrainfile):       
            print 'error: %s does not exist' % reducedtrainfile
            sys.exit(1)
        if not os.path.isfile(reducedtestfile):       
            print 'error: %s does not exist' % reducedtestfile
            sys.exit(1)

        test,train = train_and_test_model(args, train, test, reducedtrainfile, reducedtestfile, outname)
        if not crossval:
            continue

        testaucs.append([x[0] for x in test])
        trainaucs.append([x[0] for x in train])
        alltest.append(test)
        if len(test[-1]) > 4:
            testrmsds.append([x[3] for x in test])
            trainrmsds.append([x[3] for x in train])
            
        if np.mean(trainaucs) > 0:
            with open('%s.%s.finaltest' % (outprefix,m.group(1)), mode) as out:
                for (label,score) in zip(test[-1][1],test[-1][2]):
                    out.write('%f %f\n'%(label,score))
                out.write('# AUC %f\n'%test[-1][0])

        if testrmsds:
            with open('%s.%s.rmsd.finaltest' % (outprefix,m.group(1)),mode) as out:
                 for (aff,pred) in zip(test[-1][4],test[-1][5]):
                    out.write('%f %f\n'%(aff,pred))
                 out.write('# RMSD %f \n'%test[-1][3])

    if len(args.foldnums) <= 1:
        sys.exit(0)
        
    #find average, min, max AUC for last 1000 iterations
    lastiter_testaucs = []
    lastiter = 1000
    if lastiter > args.iterations: lastiter = args.iterations
    num_testaucs = lastiter/args.test_interval
    for i in xrange(len(testaucs)):
        a = testaucs[i][len(testaucs[i])-num_testaucs:]
        if a:
            lastiter_testaucs.append(a)
        
    txt = ''
    if lastiter_testaucs:
        avgAUC = np.mean(lastiter_testaucs)
        maxAUC = np.max(lastiter_testaucs)
        minAUC = np.min(lastiter_testaucs)
        txt = 'For the last %s iterations:\nmean AUC=%.2f  max AUC=%.2f  min AUC=%.2f'%(lastiter,avgAUC,maxAUC,minAUC)
    
    #average aucs, train and test

    #due to early termination length of results may not be equivalent
    testaucs = np.array(zip(*testaucs))
    trainaucs = np.array(zip(*trainaucs))
    testrmsds = np.array(zip(*testrmsds))
    trainrmsds = np.array(zip(*trainrmsds))

    with open('%s.test' % outprefix,mode) as out:
        for r in testaucs:
            out.write('%s %s\n' % (np.mean(r),' '.join([str(x) for x in r])))

    with open('%s.train' % outprefix,mode) as out:
        for r in trainaucs:
            out.write('%s %s\n' % (np.mean(r),' '.join([str(x) for x in r])))    
                        
    #make training plot
    plt.plot(trainaucs.mean(axis=1),label='Train')
    plt.plot(testaucs.mean(axis=1),label='Test')
    plt.legend(loc='best')
    plt.savefig('%s_train.pdf'%outprefix,bbox_inches='tight')
                        
    #roc curve for the last iteration - combine all tests
    n = len(testaucs)-1
    ytrue = []
    yscore = []      
    for test in alltest:
        ytrue += test[n][1]
        if test[n][2]:
            yscore += test[n][2]
    
    if len(np.unique(ytrue)) > 1:
        fpr, tpr, _ = sklearn.metrics.roc_curve(ytrue,yscore)
        auc = sklearn.metrics.roc_auc_score(ytrue,yscore)
        
        with open('%s.finaltest' % outprefix,mode) as out:
            for (label,score) in zip(ytrue,yscore):
                out.write('%f %f\n'%(label,score))
            out.write('# AUC %f\n'%auc)
            
        #make plot
        fig = plt.figure(figsize=(8,8))
        plt.plot(fpr,tpr,label='CNN (AUC=%.2f)'%(auc),linewidth=4)
        plt.legend(loc='lower right',fontsize=20)
        plt.xlabel('False Positive Rate',fontsize=22)
        plt.ylabel('True Positive Rate',fontsize=22)
        plt.axes().set_aspect('equal')
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.text(.05, -.25, txt, fontsize=22)
        plt.savefig('%s_roc.pdf'%outprefix,bbox_inches='tight')
    if len(testrmsds) > 0:
        with open('%s.rmsd.test' % outprefix,mode) as out:
            for r in testrmsds:
                out.write('%s %s\n' % (np.mean(r),' '.join([str(x) for x in r])))
    
        with open('%s.rmsd.train' % outprefix,mode) as out:
            for r in trainrmsds:
                out.write('%s %s \n' % (np.mean(r),' '.join([str(x) for x in r])))
        # training plot
        fig = plt.figure()
        plt.plot(trainrmsds.mean(axis=1),label='Train')
        plt.plot(testrmsds.mean(axis=1),label='Test')
        plt.legend(loc='best')
        plt.savefig('%s_rmsd_train.pdf'%outprefix,bbox_inches='tight')
                        
        yaffinity = []
        ypredaff = []      
        for test in alltest:
            yaffinity += list(test[-1][4])
            ypredaff += list(test[-1][5])
        yaffinity = np.array(yaffinity)
        ypredaff = np.array(ypredaff)
        yt = np.array(ytrue,dtype=np.bool)
        rmsdt = sklearn.metrics.mean_squared_error(yaffinity[yt],ypredaff[yt])
        r2t = sklearn.metrics.r2_score(yaffinity[yt],ypredaff[yt])
        
        with open('%s.rmsd.finaltest' % outprefix,mode) as out:
            for (aff,pred) in zip(yaffinity,ypredaff):
                out.write('%f %f\n'%(aff,pred))
            out.write('# RMSD,R^2 %f %f \n'%(rmsdt,r2t))
        
        #correlation plot        
        fig = plt.figure(figsize=(8,8))
        plt.plot(yaffinity[yt],ypredaff[yt],'o',label='RMSD=%.2f, R^2=%.3f (Pos)'%(rmsdt,r2t))
        plt.legend(loc='best',fontsize=20,numpoints=1)
        lo = np.min([np.min(yaffinity[yt]),np.min(ypredaff[yt])])
        hi = np.max([yaffinity[yt].max(),ypredaff[yt].max()])
        plt.xlim(lo,hi)
        plt.ylim(lo,hi)
        plt.xlabel('Experimental Affinity',fontsize=22)
        plt.ylabel('Predicted Affinity',fontsize=22)
        plt.axes().set_aspect('equal')
        plt.savefig('%s_rmsd.pdf'%outprefix,bbox_inches='tight')        
