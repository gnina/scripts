#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, re, sklearn, collections, argparse, sys, os
import sklearn.metrics
import caffe

'''Script for training a neural net model from gnina grid data.
A model template is provided along with training and test sets of the form
<prefix>[train|test][num].binmaps
Test area, as measured by AUC, is periodically assessed.   At the end graphs are made.
Default is to do dynamic stepping of learning rate, but can explore other methods.
'''
def eval_model(args, trainfile, testfile, outname):
    '''run solver for iterations steps, on the given training file,
    every testiter evaluate the roc of bothe the trainfile and the testfile
    return the full predictions for every tested iteration'''
    template = args.model
    testiter = args.test_interval
    iterations = args.iterations
    model = open(template).read().replace('TRAINFILE',trainfile)
    testmodel = model.replace('TESTFILE',testfile)
    trainmodel = model.replace('TESTFILE',trainfile) #for test on train
    
    out = open('%s.out' % outname,'w')

    pid = os.getpid()
    #very obnoxiously, python interface requires network definition to be in a file
    with open('traintest.%d.prototxt' % pid,'w') as f:
        f.write(testmodel)    
    with open('traintrain.%d.prototxt' % pid,'w') as f:
        f.write(trainmodel)
    solverf = 'solver.%d.prototxt'%pid
    solver_text = '''
    # The train/test net protocol buffer definition
    train_net: "traintest.%d.prototxt"
    test_net: "traintest.%d.prototxt"
    test_net: "traintrain.%d.prototxt"
    # The base learning rate, momentum and the weight decay of the network.
    base_lr: %f
    momentum: %f
    test_iter: 1
    test_iter: 1
    test_interval: 100000 #we will test manually, these are just here to make caffe happy
    weight_decay: %f
    # The learning rate policy
    lr_policy: "%s"
    gamma: %f
    power: %f
    display: 0
    # reproducible results
    random_seed: %d
    # The maximum number of iterations
    max_iter: %d
    snapshot_prefix: "%s"
    ''' % (pid,pid,pid, args.base_lr, args.momentum, args.weight_decay, args.lr_policy, args.gamma, args.power, args.seed, iterations,outname)
    with open(solverf,'w') as f:
        f.write(solver_text)
        
    if args.gpu >= 0:
        caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solverf) #this loads the net
    ntests = sum(1 for line in open(testfile))
    ntrains = sum(1 for line in open(trainfile))

    testvals = []
    trainvals = []
    bestauc = 0
    bestauci = 0;
    besttestauc = 0
    for i in xrange(iterations/testiter):
        solver.step(testiter)
        #evaluate test set
        y_true = []
        y_score = []
        for _ in xrange(ntests):
            res = solver.test_nets[0].forward()
            #MUST copy values out of res as it is return by ref
            y_true.append(float(res['labelout']))
            y_score.append(float(res['output'][0][1]))
            
        testauc = sklearn.metrics.roc_auc_score(y_true,y_score)
        testvals.append((testauc,y_true,y_score))
        
        if testauc > besttestauc:
            besttestauc = testauc
            solver.snapshot()
        #evaluate train set
        y_true = []
        y_score = []
        losses = []
        for _ in xrange(ntrains):
            res = solver.test_nets[1].forward()
            #MUST copy values out of res as it is return by ref
            y_true.append(float(res['labelout']))
            y_score.append(float(res['output'][0][1]))
            losses.append(float(res['loss']))
        
        trainauc = sklearn.metrics.roc_auc_score(y_true,y_score)            
        loss = np.mean(losses)
        trainvals.append((trainauc,y_true,y_score,loss))
        
        if trainauc > bestauc:
            bestauc = trainauc
            bestauci = i
            
        if args.dynamic: #check for improvement
            lr = solver.get_base_lr()
            if (i-bestauci) > args.step_when: #reduce learning rate
                lr *= args.step_reduce
                solver.set_base_lr(lr)
                bestauci = i #reset 
                bestauc = trainauc #the value too, so we can consider the recovery
            if lr < args.step_end:
                break #end early  
            
        out.write('%.4f %.4f %.6f %.6f\n'%(testauc,trainauc,loss,solver.get_base_lr()))
        out.flush()
    
    out.close()
    solver.snapshot()
    del solver #free mem
    return testvals,trainvals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train neural net on binmap data.')
    parser.add_argument('-m','--model',type=str,required=True,help="Model template. Must use TRAINFILE and TESTFILE")
    parser.add_argument('-p','--prefix',type=str,required=True,help="Prefix for training/test files: <prefix>[train|test][num].binmaps")
    parser.add_argument('-n','--number',type=int,required=False,help="Fold number to run, default is all",default=-1)
    parser.add_argument('-i','--iterations',type=int,required=False,help="Number of iterations to run,default 10,000",default=10000)
    parser.add_argument('-s','--seed',type=int,help="Random seed, default 42",default=42)
    parser.add_argument('-t','--test_interval',type=int,help="How frequently to test (iterations), default 40",default=40)
    parser.add_argument('-o','--outprefix',type=str,help="Prefix for output files, default <model>.<pid>",default='')
    parser.add_argument('-g','--gpu',type=int,help='Specify GPU to run on',default=-1)
    #parser.add_argument('-v,--verbose',action='store_true',default=False,help='Verbose output')
    parser.add_argument('--dynamic',action='store_true',default=False,help='Attempt to adjust the base_lr in response to training progress')
    parser.add_argument('--lr_policy',type=str,help="Learning policy to use. Default is inv.",default='inv')
    parser.add_argument('--step_reduce',type=float,help="Reduce the learning rate by this factor with dynamic stepping, default 0.5",default='0.5')
    parser.add_argument('--step_end',type=float,help='Terminate training if learning rate gets below this amount',default=0)
    parser.add_argument('--step_when',type=int,help="Perform a dynamic step (reduce base_lr) when training has not improved after this many test iterations, default 10",default=10)
    parser.add_argument('--base_lr',type=float,help='Initial learning rate, default 0.01',default=0.01)
    parser.add_argument('--momentum',type=float,help="Momentum parameters, default 0.9",default=0.9)
    parser.add_argument('--weight_decay',type=float,help="Weight decay, default 0.005",default=0.005)
    parser.add_argument('--gamma',type=float,help="Gamma, default 0.001",default=0.001)
    parser.add_argument('--power',type=float,help="Power, default 2",default=2)
    args = parser.parse_args()
    
    #identify all train/test pair
    if args.number >= 0:
        pairs = [('%strain%d.binmaps'%(args.prefix,args.number),'%stest%d.binmaps'%(args.prefix,args.number))]
    else:
        pairs = []
        for train in glob.glob('%strain[0-9]*.binmaps' % args.prefix):
            test = train.replace('%strain' % args.prefix,'%stest' % args.prefix)
            if not os.path.isfile(test):
                print test,' test file does not exist'
                sys.exit(1)
            pairs.append((train,test))
            
    outprefix = args.outprefix
    if outprefix == '':
        outprefix = '%s.%d' % (os.path.splitext(os.path.basename(args.model))[0],os.getpid())
    #train each Pair
    testaucs = []
    trainaucs = []
    alltest = []
    for (train,test) in pairs:
        m = re.search('%strain(\d+)'%args.prefix,train)
        outname = '%s.%s' % (outprefix,m.group(1))
        test,train = eval_model(args, train, test, outname)
        testaucs.append([x[0] for x in test])
        trainaucs.append([x[0] for x in train])
        alltest.append(test)

    #average aucs, train and test

    #due to early termination length of results may not be equivalent
    testaucs = np.array(zip(*testaucs))
    trainaucs = np.array(zip(*trainaucs))
    
    with open('%s.test' % outprefix,'w') as out:
        for r in testaucs:
            out.write('%s %s\n' % (np.mean(r),' '.join([str(x) for x in r])))

    with open('%s.train' % outprefix,'w') as out:
        for r in trainaucs:
            out.write('%s %s\n' % (np.mean(r),' '.join([str(x) for x in r])))
            
    #make training plot
    plt.plot(trainaucs.mean(axis=0),label='Train')
    plt.plot(testaucs.mean(axis=0),label='Test')
    plt.legend(loc='best')
    plt.savefig('%s_train.pdf'%outprefix,bbox_inches='tight')
                        
    #roc curve for the last iteration - combine all tests
    n = len(testaucs)-1
    ytrue = []
    yscore = []      
    for test in alltest:
        ytrue += test[n][1]
        yscore += test[n][2]
    fpr, tpr, _ = sklearn.metrics.roc_curve(ytrue,yscore)
    auc = sklearn.metrics.roc_auc_score(ytrue,yscore)
    
    with open('%s.finaltest' % outprefix,'w') as out:
        for (label,score) in zip(ytrue,yscore):
            out.write('%f %f\n'%(label,score))
        out.write('# AUC %f\n'%auc)
        
    #make plot
    plt.figure(figsize=(8,8))
    plt.plot(fpr,tpr,label='CNN (AUC=%.2f)'%auc,linewidth=4)
    plt.legend(loc='lower right',fontsize=24)
    plt.xlabel('False Positive Rate',fontsize=22)
    plt.ylabel('True Positive Rate',fontsize=22)
    plt.axes().set_aspect('equal')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig('%s_roc.pdf'%outprefix,bbox_inches='tight')
            
    
    
    
