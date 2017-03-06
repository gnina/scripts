#!/usr/bin/env python

import google.protobuf
import numpy as np
import matplotlib
from numpy import dtype
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, re, sklearn, collections, argparse, sys, os
import sklearn.metrics
import caffe
import time

'''Script for training a neural net model from gnina grid data.
A model template is provided along with training and test sets of the form
<prefix>[train|test][num].types
Test area, as measured by AUC, is periodically assessed.   At the end graphs are made.
Default is to do dynamic stepping of learning rate, but can explore other methods.
'''
def eval_model(args, trainfile, testfile, reducedtrainfile, reducedtestfile, outname):
    '''run solver for iterations steps, on the given training file,
    every testiter evaluate the roc of bothe the trainfile and the testfile
    return the full predictions for every tested iteration'''
    template = args.model
    testiter = args.test_interval
    iterations = args.iterations
    
    if testiter > iterations: #need to test once
        testiter = iterations
    model = open(template).read().replace('TRAINFILE',trainfile)
    testmodel = model.replace('TESTFILE',testfile)
    trainmodel = model.replace('TESTFILE',trainfile) #for test on train
    if reducedtrainfile != '':        
        reducedtrainmodel = model.replace('TESTFILE', reducedtrainfile)
    else:
        reducedtrainmodel = trainmodel
        reducedtrainfile = trainfile
    if reducedtestfile != '':        
        reducedtestmodel = model.replace('TESTFILE', reducedtestfile)
    else:
        reducedtestmodel = testmodel
        reducedtestfile = testfile
        
    if args.avg_rotations:
        rotations = 24 
        index=testmodel.find(testfile) #add 'rotate = 24' to testmodels if not already there
        endindex=testmodel.find('layer', index)
        rot = testmodel.find("rotate:", index, endindex)
        if rot == -1:
            index = testmodel.find('balanced:', index, endindex)
            index = testmodel.find('\n', index, endindex)
            testmodel = testmodel[:index+1] + '    rotate: %d'% rotations + testmodel[index:]
        
        index=reducedtestmodel.find(reducedtestfile)
        endindex=reducedtestmodel.find('layer', index)
        rot = reducedtestmodel.find("rotate:", index, endindex)
        if rot == -1:
            index = reducedtestmodel.find('balanced:', index, endindex)
            index = reducedtestmodel.find('\n', index, endindex)
            reducedtestmodel = reducedtestmodel[:index+1] + '    rotate: %d' %rotations + reducedtestmodel[index:]
		    

    mode = 'w'
    if args.cont:
        mode = 'a'    
        modelname = '%s_iter_%d.caffemodel' % (outname,args.cont)
        solvername = '%s_iter_%d.solverstate' % (outname,args.cont)
        
    out = open('%s.out' % outname,mode,0)

    pid = os.getpid()
    #very obnoxiously, python interface requires network definition to be in a file
    testproto = 'traintest.%d.prototxt' % pid
    trainproto = 'traintrain.%d.prototxt' % pid
    reducedtestproto = 'trainreducedtest.%d.prototxt' % pid
    reducedtrainproto = 'trainreducedtrain.%d.prototxt' % pid
    with open(testproto,'w') as f:
        f.write(testmodel)    
    with open(trainproto,'w') as f:
        f.write(trainmodel)
    with open(reducedtestproto, 'w') as f:
        f.write(reducedtestmodel) 
    with open(reducedtrainproto, 'w') as f:
        f.write(reducedtrainmodel) 
    solverf = 'solver.%d.prototxt'%pid
    solver_text = '''
    # The train/test net protocol buffer definition
    train_net: "traintest.%d.prototxt"
    test_net: "traintest.%d.prototxt"
    test_net: "traintrain.%d.prototxt"
    test_net: "trainreducedtest.%d.prototxt"
    test_net: "trainreducedtrain.%d.prototxt"
    # The base learning rate, momentum and the weight decay of the network.
    type: "%s"
    base_lr: %f
    momentum: %f
    test_iter: 1
    test_iter: 1
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
    ''' % (pid,pid,pid,pid,pid, args.solver,args.base_lr, args.momentum, args.weight_decay, args.lr_policy, args.gamma, args.power, args.seed, iterations+args.cont,outname)
    with open(solverf,'w') as f:
        f.write(solver_text)
        
    if args.gpu >= 0:
        caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    
    solver = caffe.get_solver(solverf)
        
    if args.cont:
        solver.restore(solvername)
        solver.testall() #link testnets to train net

    if args.weights:
        solver.net.copy_from(args.weights)
        
    ntests = sum(1 for line in open(reducedtestfile))
    ntrains = sum(1 for line in open(reducedtrainfile))

    testvals = []
    trainvals = []
    bestauc = 0
    bestauci = 0;
    besttestauc = 0
    testauc = 0
    trainauc = 0
    loss = 0
    #print "cnts %d,%d" % (ntrains,ntests)
    for i in xrange(iterations/testiter):
        start = time.time()
        solver.step(testiter)

        print "Train time: %f" % (time.time()-start)

        start = time.time()
        #evaluate test set
        if i == (iterations/testiter)-1 and args.reduced:
            testnet = solver.test_nets[0]
            ntests = sum(1 for line in open(testfile))
        else:
            testnet = solver.test_nets[2]
        y_true = []
        y_score = []
        y_scores = [[] for _ in xrange(ntests)]
        y_affinity = []
        y_predaff = []
        y_predaffs = [[] for _ in xrange(ntests)]
        for x in xrange(ntests):
            res = testnet.forward()
            #MUST copy values out of res as it is return by ref
            y_true.append(float(res['labelout']))
            if 'output' in res:
                y_scores[x].append(float(res['output'][0][1])) 
            else:
                y_scores[x].append(0)
            if 'affout' in res:
                y_affinity.append(float(res['affout']))
                y_predaffs[x].append(float(res['predaff']))
        if args.avg_rotations:
            for _ in xrange(rotations-1):
                print ntests #check if ntests is correct
                for x in xrange(ntests):
                    res = testnet.forward()
                    if 'output' in res:
                        yt = float(res['labelout'])
                        if yt != y_true[x]:
                            print "%dERROR: %f,y_true: %f" %(x,yt, y_true[x]) #sanity check
                        y_scores[x].append(float(res['output'][0][1]))
                    if y_affinity:
                        y_predaffs[x].append(float(res['predaff']))
            #average the scores for the 24 rotations
            for x in xrange(ntests):
                if y_scores:
                    y_score.append(np.mean(y_scores[x]))
                if y_affinity:
                    y_predaff.append(np.mean(y_predaffs[x]))
        else:
            y_score = [row[0] if len(row) > 0 else [] for row in y_scores[0:]]
            y_predaff = [row[0] if len(row) > 0 else [] for row in y_predaffs]
            
        print "Test time: %f" % (time.time()-start)
        if len(np.unique(y_true)) > 1:
            testauc = sklearn.metrics.roc_auc_score(y_true,y_score)
        
        if y_affinity:
            y_predaff = np.array(y_predaff)
            yt = np.array(y_true,np.bool)
            y_affinity = np.array(y_affinity)
            testrmsd = sklearn.metrics.mean_squared_error(y_affinity[yt],y_predaff[yt])
            testvals.append((testauc,y_true,y_score,testrmsd,y_affinity,y_predaff))              
        else:
            testvals.append((testauc,y_true,y_score))
        
        print "Test eval: %f s" % (time.time()-start)
        
        if testauc > besttestauc:
            besttestauc = testauc
            if args.keep_best:
                solver.snapshot() #a bit too much - gigabytes of data
        
        start = time.time()
        #evaluate train set
        start = time.time()
        if i == (iterations/testiter)-1 and args.reduced:
            testnet = solver.test_nets[1]
            ntrains= sum(1 for line in open(trainfile))
        else:
            testnet = solver.test_nets[3]
        y_true = []
        y_score = []
        y_affinity = []
        y_predaff = []
        losses = []
        for x in xrange(ntrains):
            res = testnet.forward()            
            #MUST copy values out of res as it is return by ref
            y_true.append(float(res['labelout']))
            if 'output' in res:
                y_score.append(float(res['output'][0][1]))
            else:
                y_score.append(0)
            if 'loss' in res:
                losses.append(float(res['loss']))
            else:
                losses.append(0)
            if 'affout' in res:
                y_affinity.append(float(res['affout']))
                y_predaff.append(float(res['predaff']))
        
        print "Test train time: %f" % (time.time()-start)
        if len(np.unique(y_true)) > 1:
            trainauc = sklearn.metrics.roc_auc_score(y_true,y_score)            
        loss = np.mean(losses)
        
        if y_affinity:
            y_predaff = np.array(y_predaff)
            y_affinity = np.array(y_affinity)
            trainrmsd = sklearn.metrics.mean_squared_error(y_affinity[yt],y_predaff[yt])                    
            if y_score:
                yt = np.array(y_true,np.bool)
                trainvals.append((trainauc,y_true,y_score,loss,trainrmsd,y_affinity,y_predaff))
            else:
                trainvals.append((0,[],[],0,trainrmsd,y_affinity,y_predaff))
        else:
            trainvals.append((trainauc,y_true,y_score,loss))

        print "Train eval: %f s" % (time.time()-start)
        
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
            
        out.write('%.4f %.4f %.6f %.6f'%(testauc,trainauc,loss,solver.get_base_lr()))
        if len(y_affinity):
            out.write(' %.4f %.4f' % (testrmsd,trainrmsd))
        out.write('\n')
        out.flush()
    
    out.close()
    solver.snapshot()
    del solver #free mem
    
    if not args.keep:
        os.remove(solverf)
        os.remove(testproto)
        os.remove(trainproto)
        os.remove(reducedtestproto)
        os.remove(reducedtrainproto)
    return testvals,trainvals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('-m','--model',type=str,required=True,help="Model template. Must use TRAINFILE and TESTFILE")
    parser.add_argument('-p','--prefix',type=str,required=True,help="Prefix for training/test files: <prefix>[train|test][num].types")
    parser.add_argument('-n','--number',type=int,required=False,help="Fold number to run, default is all",default=-1)
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
    args = parser.parse_args()
    
    #identify all train/test pair
    if args.number >= 0:
        pairs = [('%strain%d.types'%(args.prefix,args.number),'%stest%d.types'%(args.prefix,args.number))]
    else:
        pairs = []
        for train in glob.glob('%strain[0-9]*.types' % args.prefix):
            test = train.replace('%strain' % args.prefix,'%stest' % args.prefix)
            if not os.path.isfile(test):
                print test,' test file does not exist'
                sys.exit(1)
            pairs.append((train,test))

    if len(pairs) == 0:
        print "Missing train/test files"
        sys.exit(1)
                
    outprefix = args.outprefix
    if outprefix == '':
        outprefix = '%s.%d' % (os.path.splitext(os.path.basename(args.model))[0],os.getpid())
        
    mode = 'w'
    if args.cont:
        mode = 'a'
        
    #train each Pair
    testaucs = []
    trainaucs = []
    testrmsds = []
    trainrmsds = []
    alltest = []
    for (train,test) in pairs:
        m = re.search('%strain(\d+)'%args.prefix,train)
        outname = '%s.%s' % (outprefix,m.group(1))

        reducedtrainfile = train.replace('%strain' % args.prefix,'%s_reducedtrain' % args.prefix)
        reducedtestfile = train.replace('%strain' % args.prefix,'%s_reducedtest' % args.prefix)
        if not os.path.isfile(reducedtestfile):       
            reducedtestfile = '' 
        if not os.path.isfile(reducedtrainfile):       
            reducedtrainfile = ''

        test,train = eval_model(args, train, test, reducedtrainfile, reducedtestfile, outname)
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


    if args.number >= 0:
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
