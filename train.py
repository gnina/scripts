#!/usr/bin/env python

import numpy as np
import matplotlib
from numpy import dtype
from scipy.stats._continuous_distns import foldcauchy
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, re, sklearn, collections, argparse, sys, os
import sklearn.metrics
import caffe
from caffe.proto.caffe_pb2 import NetParameter, SolverParameter
import google.protobuf.text_format as prototxt
import time
import psutil
import cPickle, signal
from combine_fold_results import write_results_file, combine_fold_results


# class based on: http://stackoverflow.com/a/21919644/487556
# this tries to protecta critical section from being interrupted
# (obviously can't do anything with SIGKILL)
class DelayedInterrupt(object):
    def __init__(self, signals):
        if not isinstance(signals, list) and not isinstance(signals, tuple):
            signals = [signals]
        self.sigs = signals        

    def __enter__(self):
        self.signal_received = {}
        self.old_handlers = {}
        for sig in self.sigs:
            self.signal_received[sig] = False
            self.old_handlers[sig] = signal.getsignal(sig)
            def handler(s, frame):
                self.signal_received[sig] = (s, frame)
                # Note: in Python 3.5, you can use signal.Signals(sig).name
                logging.info('Signal %s received. Delaying KeyboardInterrupt.' % sig)
            self.old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)

    def __exit__(self, type, value, traceback):
        for sig in self.sigs:
            signal.signal(sig, self.old_handlers[sig])
            if self.signal_received[sig] and self.old_handlers[sig]:
                self.old_handlers[sig](*self.signal_received[sig])
                
'''Script for training a neural net model from gnina grid data.
A model template is provided along with training and test sets of the form
<prefix>[train|test][num].types
Test accuracy, as measured by AUC, is periodically assessed.
At the end graphs are made.'''


def write_model_file(model_file, template_file, train_file, test_file, root_folder, avg_rotations=False,
                     train_file2=None, ratio=None, root_folder2=None, test_root_folder=None):
    '''Writes a model prototxt file based on a provided template file
    with certain placeholders replaced in each MolGridDataLayer.
    For the source parameter, "TRAINFILE" is replaced with train_file
    and "TESTFILE" is replaced with test_file.
    For the root_folder parameter, "DATA_ROOT" is replaced with root_folder,
    unless the layer is TEST phase and test_root_folder is provided,
    then it is replaced with test_root_folder.
    For the source2 parameter, "TRAINFILE2" is replaced with train_file2,
    and in the same layer the source_ratio parameter is set to ratio.
    For the root_folder2 parameter, "DATA_ROOT2" is replaced with root_folder2.
    If the avg_rotations argument is set and the layer is TEST phase,
    the rotate parameter is set to 24.'''
    netparam = NetParameter()
    with open(template_file, 'r') as f:
        prototxt.Merge(f.read(), netparam)
    for layer in netparam.layer:
        if layer.type == "NDimData":
            param = layer.ndim_data_param
        elif layer.type == "MolGridData":
            param = layer.molgrid_data_param
        else:
            continue
        if param.source == 'TRAINFILE':
            param.source = train_file
        if param.source == 'TESTFILE':
            param.source = test_file
        if param.root_folder == 'DATA_ROOT':
            if test_root_folder and 'TEST' in str(layer):
                param.root_folder = test_root_folder
            else:
                param.root_folder = root_folder
        if train_file2 and param.source2 == 'TRAINFILE2':
            param.source2 = train_file2
            param.source_ratio = ratio
        if root_folder2 and param.root_folder2 == 'DATA_ROOT2':
            param.root_folder2 = root_folder2
        if avg_rotations and 'TEST' in str(layer):
            param.rotate = 24 #TODO axial rotations aren't working
            #layer.molgrid_data_param.random_rotation = True
    with open(model_file, 'w') as f:
        f.write(str(netparam))


def write_solver_file(solver_file, train_model, test_models, type, base_lr, momentum, weight_decay,
                      lr_policy, gamma, power, random_seed, max_iter, clip_gradients, snapshot_prefix):
    '''Writes a solver prototxt file with parameters set to the
    corresponding argument values. In particular, the train_net
    parameter is set to train_model, and a test_net parameter is
    added for each of test_models, which should be a list.'''
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
    if clip_gradients > 0:
        param.clip_gradients = clip_gradients
    param.snapshot_prefix = snapshot_prefix
    print "WRITING",solver_file
    with open(solver_file,'w') as f:
        f.write(str(param))


class Namespace():
    '''Simple object with better readability than dict'''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def evaluate_test_net(test_net, n_tests, n_rotations, offset):
    '''Evaluate a test network and return the results. The number of
    examples in the file the test_net reads from must equal n_tests,
    otherwise output will be misaligned. Can optionally take the average
    of multiple rotations of each example. Offset is the index into
    the test file that will be the first example in the next batch.
    Net parameters should be set so that data access is sequential.'''

    #evaluate each example with each rotation
    y_true     = [-1 for _ in xrange(n_tests)]
    y_scores   = [[] for _ in xrange(n_tests)]
    y_affinity = [-1 for _ in xrange(n_tests)]
    y_predaffs = [[] for _ in xrange(n_tests)]
    rmsd_true = [-1 for _ in xrange(n_tests)]
    rmsd_pred = [[] for _ in xrange(n_tests)]
    
    losses = []

    rmsd_true_blob = test_net.blobs.get('rmsd_true')
    rmsd_pred_blob = test_net.blobs.get('rmsd_pred')
    rmsd_loss_blob = test_net.blobs.get('rmsd_loss')

    res = None
    for r in xrange(n_rotations):
        for x in xrange(n_tests):
            x = (x + offset) % n_tests

            if not res or i >= batch_size:
                res = test_net.forward()
                if 'output' in res:
                    batch_size = res['output'].shape[0]
                elif 'affout' in res:
                    batch_size = res['affout'].shape[0]
                else:                    
                    batch_size = res['label'].shape[0]
                i = 0

            if 'labelout' in res:
                if r == 0:
                    y_true[x] = float(res['labelout'][i])
                else:
                    assert y_true[x] == res['labelout'][i] #sanity check

            if 'output' in res:
                y_scores[x].append(float(res['output'][i][1]))

            if 'affout' in res:
                if r == 0:
                    y_affinity[x] = float(res['affout'][i])
                else:
                    assert y_affinity[x] == res['affout'][i] #sanity check

            if 'predaff' in res:
                y_predaffs[x].append(float(res['predaff'][i]))
            if 'loss' in res:
                losses.append(float(res['loss']))                
                
            if rmsd_true_blob:
                if r == 0:
                    rmsd_true[x] = float(rmsd_true_blob.data[i])
            if rmsd_pred_blob:
                rmsd_pred[x].append(float(rmsd_pred_blob.data[i]))
                
            i += 1

    #get index of test example that will be at start of next batch
    offset = (x + 1 + batch_size - i) % n_tests

    result = Namespace(auc=None, y_true=y_true, y_score=[], loss=None,
                       rmsd=None, y_aff=y_affinity, rmsd_true=rmsd_true,
                       rmsd_pred=[],rmsd_rmse=None, y_predaff=[])

    #average the scores from each rotation
    if any(y_scores):
        for x in xrange(n_tests):
            result.y_score.append(np.mean(y_scores[x]))

    if any(y_predaffs):
        for x in range(n_tests):
            result.y_predaff.append(np.mean(y_predaffs[x]))

    if any(rmsd_pred):
        for x in range(n_tests):
            result.rmsd_pred.append(np.mean(rmsd_pred[x]))

                        
    #compute auc
    if result.y_true and result.y_score:
        if len(np.unique(result.y_true)) > 1:
            result.auc = sklearn.metrics.roc_auc_score(result.y_true, result.y_score)
        else: # may be evaluating all crystal poses?
            print "Warning: only one unique label"
            result.auc = 1.0

    #compute mean squared error (rmsd) of affinity (for actives only)
    if result.y_aff and result.y_predaff:
        y_predaff_true = np.array(result.y_predaff)[np.array(result.y_aff)>0]#filter_actives(result.y_predaff, result.y_true)
        y_aff_true = np.array(result.y_aff)[np.array(result.y_aff)>0]#filter_actives(result.y_aff, result.y_true)
            
        result.rmsd = np.sqrt(sklearn.metrics.mean_squared_error(y_aff_true, y_predaff_true))

    if any(rmsd_pred):
        result.rmsd_rmse = np.sqrt(sklearn.metrics.mean_squared_error(result.rmsd_pred,result.rmsd_true))
        
    #compute mean loss
    if losses:
        result.loss = np.mean(losses)

    return result, offset


def count_lines(file):
    return sum(1 for line in open(file, 'r'))


def train_and_test_model(args, files, outname, cont=0):
    '''Train caffe model for iterations steps using provided model template
    and training file(s), and every test_interval iterations evaluate each
    of the train and test files. Return AUC (and RMSD, if affinity model)
    for every test iteration, and also the labels and predictions for the
    final test iteration. If cont > 0, assumes the presence of a saved 
    caffemodel at that iteration.'''
    
    #helper functions
    def freemem():
        '''Free intermediate blobs from all networks.  These will be reallocated as needed.'''
        net = solver.net
        if net.clearblobs:
            #solver will need values in output blobs
            for (bname, blob) in net.blobs.iteritems():
                if bname not in net.outputs:
                    blob.clear()
            for k in test_nets.iterkeys():
                test_nets[k][0].clearblobs()

        

    def update_from_result(name, test, result):
        '''Put results into test/train structure'''
        test.y_true = result.y_true
        test.y_score = result.y_score
        test.y_aff = result.y_aff
        test.y_predaff = result.y_predaff
        test.rmsd_true = result.rmsd_true
        test.rmsd_pred = result.rmsd_pred
        if result.auc is not None:
            print "%s AUC: %f" % (name,result.auc)
            test.aucs.append(result.auc)
        if result.loss:
            print "%s loss: %f" % (name,result.loss)
            test.losses.append(result.loss)
        if result.rmsd is not None:
            print "%s RMSD: %f" % (name,result.rmsd)
            test.rmsds.append(result.rmsd)
        if result.rmsd_rmse is not None:
            print "%s rmsd_rmse: %f" % (name,result.rmsd_rmse)
            test.rmsd_rmses.append(result.rmsd_rmse)
                
                    
    template = args.model
    test_interval = args.test_interval
    iterations = args.iterations-cont
    training = not args.test_only

    if args.test_only:
        test_interval = iterations = 1
    elif test_interval > iterations: #need to test once
        test_interval = iterations

    if args.avg_rotations:
        rotations = 24
    else:
        rotations = 1

    pid = os.getpid()

    #write model prototxts (for each file to test)
    test_on_train = files['test'] == files['train']
    test_models = ['traintest.%d.prototxt' % pid]
    test_files = [files['test']]
    test_roots = [args.data_root] #which data_root to use
    if args.reduced:
        test_models += ['trainreducedtest.%d.prototxt' % pid]
        test_files += [files['reduced_test']]
        test_roots += [args.data_root]
    if args.prefix2:
        test_models += ['traintest2.%d.prototxt' % pid]
        test_files += [files['test2']]
        test_roots += [args.data_root2]
        if args.reduced:
            test_models += ['trainreducedtest2.%d.prototxt' % pid]
            test_files += [files['reduced_test2']]
            test_roots += [args.data_root2]
    if not test_on_train:
        test_models += ['traintrain.%d.prototxt' % pid]
        test_files += [files['train']]
        test_roots += [args.data_root]
        if args.reduced:
            test_models += ['trainreducedtrain.%d.prototxt' % pid]
            test_files += [files['reduced_train']]
            test_roots += [args.data_root]
        if args.prefix2:
            test_models += ['traintrain2.%d.prototxt' % pid]
            test_files += [files['train2']]
            test_roots += [args.data_root2]
            if args.reduced:
                test_models += ['trainreducedtrain2.%d.prototxt' % pid]
                test_files += [files['reduced_train2']]
                test_roots += [args.data_root2]

    for test_model, test_file, test_root in zip(test_models, test_files, test_roots):
        if args.prefix2:
            write_model_file(test_model, template, files['train'], test_file, args.data_root, args.avg_rotations,
                             files['train2'], args.data_ratio, args.data_root2, test_root)
        else:
            write_model_file(test_model, template, files['train'], test_file, args.data_root, args.avg_rotations)


    #initialize variables
    train = Namespace(aucs=[], y_true=[], y_score=[], losses=[], rmsds=[], y_aff=[], y_predaff=[],rmsd_rmses=[])
    if not test_on_train:
        test = Namespace(aucs=[], y_true=[], y_score=[], losses=[], rmsds=[], y_aff=[], y_predaff=[],rmsd_rmses=[])
    else:
        test = train
    if args.prefix2:
        train2 = Namespace(aucs=[], y_true=[], y_score=[], losses=[], rmsds=[], y_aff=[], y_predaff=[])
        if not test_on_train:
            test2 = Namespace(aucs=[], y_true=[], y_score=[], losses=[], rmsds=[], y_aff=[], y_predaff=[])
        else:
            test2 = train2

    #also keep track of best test and train aucs
    best_train_interval = cont
    
    bests = {'test_auc': np.inf,
        'train_loss': np.inf, \
        'test_rmsd': np.inf, \
        'train_rmsd': np.inf, \
        'test_rmsd_rmse': np.inf, \
        'train_rmsd_rmse': np.inf}      
    
    train_rmsd = np.inf
    test_rmsd = np.inf
    train_rmsd_rmse = np.inf
    test_rmsd_rmse = np.inf
    step_reduce_cnt = 0
    i_time_avg = 0
    original_lr = args.base_lr    

    #write solver prototxt
    solverf = 'solver.%d.prototxt' % pid
    write_solver_file(solverf, test_models[0], test_models, args.solver, args.base_lr, args.momentum, args.weight_decay,
                      args.lr_policy, args.gamma, args.power, args.seed, iterations+cont, args.clip_gradients, outname)

    #set up solver in caffe
    if args.gpu >= 0:
        caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solverf)
    
                    
    if cont:
        solvername = '%s_iter_%d.solverstate' % (outname, cont)
        check_file_exists(solvername)
        solver.restore(solvername)
        solver.testall() #link testnets to train net
            
    if args.checkpoint:
        checkname = '%s.CHECKPOINT'%outname
        if os.path.exists(checkname):
            print checkname
            checkdata = cPickle.load(open(checkname))
            (dontremove, training, prevsnap,train,test,bests,best_train_interval,prevlr, step_reduce_cnt) = checkdata
            
            if not training:
                print "Fold %s already completed"%outname
                return test, train

            print "Restoring",prevsnap

            solver.restore(prevsnap)
            print "Testall"
            solver.testall()            
            solver.set_base_lr(prevlr) #this isn't saved in solver state!
            #figure out iteration 
            m = re.search(r'_iter_(\d+)\.solverstate',prevsnap)
            cont = int(m.group(1))
            iterations = args.iterations-cont
            print "Continuing checkpoint from",cont
                
    if args.weights:
        check_file_exists(args.weights)
        solver.net.copy_from(args.weights) #TODO this doesn't actually set the necessary weights...
                
    test_nets = {}
    for key, test_file in files.items():
        idx = test_files.index(test_file)
        test_nets[key] = solver.test_nets[idx], count_lines(test_file), 0

    if training: #outfile is training progress, don't write if we're not training
        outfile = '%s.out' % outname
        out = open(outfile, 'a' if cont else 'w', 0) #unbuffered


    last_test = False # indicator we should test full set
    for i in xrange(iterations/test_interval):
        if i == (int(iterations/test_interval) - 1):
            last_test = True

        i_start = start = time.time()
        keepsnap = False
        if training:
            #train
            solver.step(test_interval)
            print "Iteration %d" % (cont + (i+1)*test_interval)
            print "Train time: %f" % (time.time()-start)

        if not test_on_train:
            #evaluate test set
            start = time.time()
            if args.reduced and not last_test:
                key = 'reduced_test'
            else:
                key = 'test'
            test_net, n_tests, offset = test_nets[key]
            freemem()
            result, offset = evaluate_test_net(test_net, n_tests, rotations, offset)
            test_nets[key] = test_net, n_tests, offset  #why doing this?
            print "Eval test time: %f" % (time.time()-start)

            update_from_result("Test", test, result)

            if args.prefix2:
                #evaluate test set 2
                start = time.time()
                if args.reduced and not last_test:
                    key = 'reduced_test2'
                else:
                    key = 'test2'
                test_net, n_tests, offset = test_nets[key]
                freemem()
                result, offset = evaluate_test_net(test_net, n_tests, rotations, offset)
                test_nets[key] = test_net, n_tests, offset
                print "Eval test2 time: %f" % (time.time()-start)

                update_from_result("Test2", test2, result)

        #evaluate train set
        start = time.time()
        if args.reduced and not last_test:
            key = 'reduced_train'
        else:
            key = 'train'
        test_net, n_tests, offset = test_nets[key]
        freemem()
        result, offset = evaluate_test_net(test_net, n_tests, rotations, offset)
        test_nets[key] = test_net, n_tests, offset
        print "Eval train time: %f" % (time.time()-start)

        update_from_result("Train", train, result)

        if args.prefix2:
            #evaluate train set 2
            start = time.time()
            if args.reduced and not last_test:
                key = 'reduced_train2'
            else:
                key = 'train2'
            test_net, n_tests, offset = test_nets[key]
            freemem()
            result, offset = evaluate_test_net(test_net, n_tests, rotations, offset)
            test_nets[key] = test_net, n_tests, offset
            print "Eval train2 time: %f" % (time.time()-start)

            if i > 0 and not (args.reduced and last_test): #check alignment
                assert np.all(result.y_true == train2.y_true)
                assert np.all(result.y_aff == train2.y_aff)

            update_from_result("Train2", train2, result)            

        if training:
            row = []            

            #check for improvement
            if result.auc is not None:
                test_auc = test.aucs[-1]
                train_auc = train.aucs[-1]
                train_loss = train.losses[-1]
                row += [test_auc,train_auc,train_loss]
                if test_auc > bests['test_auc']:
                    bests['test_auc'] = test_auc
                    if args.keep_best:
                        keepsnap = True
                        solver.snapshot() #a bit too much - gigabytes of data
                if train_loss < bests['train_loss']:
                    bests['train_loss'] = train_loss
                    best_train_interval = i
                    
            row += [solver.get_base_lr()]                    
            #check for rmsd improvement
            if result.rmsd is not None:
                test_rmsd = test.rmsds[-1]
                train_rmsd = train.rmsds[-1]
                if test_rmsd < bests['test_rmsd']:
                    bests['test_rmsd'] = test_rmsd
                    if args.keep_best:
                        keepsnap = True
                        solver.snapshot() #a bit too much - gigabytes of data     
                        
                if train_rmsd < bests['train_rmsd']:
                    bests['train_rmsd'] = train_rmsd
                    best_train_interval = i #note updated for both pose and aff    
                row += [test_rmsd, train_rmsd]
                    
            #check for rmse improvement
            if result.rmsd_rmse is not None:
                test_rmsd_rmse = test.rmsd_rmses[-1]
                train_rmsd_rmse = train.rmsd_rmses[-1]
                if test_rmsd_rmse < bests['test_rmsd_rmse']:
                    bests['test_rmsd_rmse'] = test_rmsd_rmse
                    if args.keep_best:
                        keepsnap = True
                        solver.snapshot() #a bit too much - gigabytes of data  
                row += [test_rmsd_rmse, train_rmsd_rmse]                            
                                                 
            if args.prefix2:  #blah
                if result.auc:
                    test2_auc = test2.aucs[-1]
                    train2_auc = train2.aucs[-1]
                    train2_loss = train2.losses[-1]
                if result.rmsd:
                    test2_rmsd = test2.rmsds[-1]
                    train2_rmsd = train2.rmsds[-1]
                if result.auc is not None:
                    row += [test2_auc, train2_auc, train2_loss]
                if result.rmsd is not None:
                    row += [test2_rmsd, train2_rmsd]
                
            #write out evaluation results                
            out.write(' '.join('%.6f' % x for x in row) + '\n')
            out.flush()

            #check for a stuck network (same prediction for everything)
            if len(result.y_score) > 1 and len(np.unique(result.y_score)) == 1:
                print "Identical scores in test, bailing early"
                break
            if len(result.y_predaff) > 1 and len(np.unique(result.y_predaff)) == 1:
                print "Identical affinities in test, bailing early"
                break
            if len(result.rmsd_pred) and len(np.unique(result.rmsd_pred)) == 1:
                print "Identical rmsd rmses in test, bailing early"
                break
                
            #update learning rate if necessary
            if args.dynamic:
                lr = solver.get_base_lr()
                if (i-best_train_interval) > args.step_when: #reduce learning rate
                    lr *= args.step_reduce
                    solver.set_base_lr(lr)
                    best_train_interval = i #reset
                    step_reduce_cnt += 1
                    
                if step_reduce_cnt > args.step_end_cnt or lr < args.step_end:
                    #end early, but run full test if needed
                    keepsnap = True
                    if args.reduced:
                        last_test = True
                    else:
                        break
            elif args.cyclic:
                lrs = [original_lr*1.5, original_lr*1.25, original_lr, original_lr*0.75, original_lr*0.5]
                indexes = [0, 1, 2, 3, 4, 3, 2, 1]
                lr = lrs[indexes[i%len(indexes)]]
                solver.set_base_lr(lr) 

        #track avg time per loop
        i_time = time.time()-i_start
        i_time_avg = (i*i_time_avg + i_time)/(i+1)
        i_left = iterations/test_interval - (i+1)
        time_left = i_time_avg * i_left
        time_str = time.strftime('%H:%M:%S', time.gmtime(time_left))
        print "Loop time: %f (%s left)" % (i_time, time_str)

        mem = psutil.Process(os.getpid()).memory_info().rss
        freemem()
        print "Memory usage: %.3fgb (%d)" % (mem/1073741824., mem)
        
        if args.checkpoint:
            snapname = solver.snapshot()
            snapname = snapname.replace('caffemodel','solverstate')

            checkname = '%s.CHECKPOINT'%outname
            #read previous snap
            if os.path.exists(checkname):
              (dontremove,_,prevsnap) = cPickle.load(open(checkname))[:3]
            else:
              dontremove = True
              prevsnap = None

            with DelayedInterrupt([signal.SIGTERM, signal.SIGINT]):
                #write new snap
                checkout = open(checkname,'w')         
                cPickle.dump((keepsnap, training, snapname,train,test,bests,best_train_interval,solver.get_base_lr(), step_reduce_cnt), checkout)
                checkout.flush()
                checkout.close()
                if prevsnap != snapname: #not sure why this would happen, but be on the safe side
                    try:
                        if not dontremove:
                            print "Removing",prevsnap
                            os.remove(prevsnap)
                            prevsnap = prevsnap.replace('solverstate','caffemodel')
                            os.remove(prevsnap)
                    except Exception as e:
                        print e
        
        if last_test:
            if training: # we indicated we are done, but still need last test
                training = False
            else: #training is false, we've done the last test
                break
    if training:
        out.close()
        solver.snapshot()

    if not args.keep:
        print "REMOVING",solverf
        os.remove(solverf)
        for test_model in test_models:
            print "REMOVING",test_model
            os.remove(test_model)

    if args.prefix2:
        return test, train, test2, train2
    else:
        return test, train

def parse_args(argv=None):
    '''Return argument namespace and commandline'''
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('-m','--model',type=str,required=True,help="Model template. Must use TRAINFILE and TESTFILE")
    parser.add_argument('-p','--prefix',type=str,required=True,help="Prefix for training/test files: <prefix>[train|test][num].types")
    parser.add_argument('-d','--data_root',type=str,required=False,help="Root folder for relative paths in train/test files",default='')
    parser.add_argument('-n','--foldnums',type=str,required=False,help="Fold numbers to run, default is to determine using glob",default=None)
    parser.add_argument('-a','--allfolds',action='store_true',required=False,help="Train and test file with all data folds, <prefix>.types",default=False)
    parser.add_argument('-i','--iterations',type=int,required=False,help="Number of iterations to run,default 10,000",default=10000)
    parser.add_argument('-s','--seed',type=int,help="Random seed, default 42",default=42)
    parser.add_argument('-t','--test_interval',type=int,help="How frequently to test (iterations), default 40",default=40)
    parser.add_argument('-o','--outprefix',type=str,help="Prefix for output files, default <model>.<pid>",default='')
    parser.add_argument('-g','--gpu',type=int,help='Specify GPU to run on',default=-1)
    parser.add_argument('-c','--cont',type=int,help='Continue a previous simulation from the provided iteration (snapshot must exist)',default=0)
    parser.add_argument('-k','--keep',action='store_true',default=False,help="Don't delete prototxt files")
    parser.add_argument('-r', '--reduced', action='store_true',default=False,help="Use a reduced file for model evaluation if exists(<prefix>[reducedtrain|reducedtest][num].types)")
    parser.add_argument('--avg_rotations', action='store_true',default=False, help="Use the average of the testfile's 24 rotations in its evaluation results")
    parser.add_argument('--checkpoint', action='store_true',default=False,help="Enable automatic checkpointing")
    #parser.add_argument('-v,--verbose',action='store_true',default=False,help='Verbose output')
    parser.add_argument('--keep_best',action='store_true',default=False,help='Store snapshots everytime test AUC improves')
    parser.add_argument('--dynamic',action='store_true',default=False,help='Attempt to adjust the base_lr in response to training progress')
    parser.add_argument('--cyclic',action='store_true',default=False,help='Vary base_lr in range of values: 0.015 to 0.001')
    parser.add_argument('--solver',type=str,help="Solver type. Default is SGD",default='SGD')
    parser.add_argument('--lr_policy',type=str,help="Learning policy to use. Default is inv.",default='inv')
    parser.add_argument('--step_reduce',type=float,help="Reduce the learning rate by this factor with dynamic stepping, default 0.1",default='0.1')
    parser.add_argument('--step_end',type=float,help='Terminate training if learning rate gets below this amount',default=0)
    parser.add_argument('--step_end_cnt',type=float,help='Terminate training after this many lr reductions',default=3)
    parser.add_argument('--step_when',type=int,help="Perform a dynamic step (reduce base_lr) when training has not improved after this many test iterations, default 5",default=5)
    parser.add_argument('--base_lr',type=float,help='Initial learning rate, default 0.01',default=0.01)
    parser.add_argument('--momentum',type=float,help="Momentum parameters, default 0.9",default=0.9)
    parser.add_argument('--weight_decay',type=float,help="Weight decay, default 0.001",default=0.001)
    parser.add_argument('--gamma',type=float,help="Gamma, default 0.001",default=0.001)
    parser.add_argument('--power',type=float,help="Power, default 1",default=1)
    parser.add_argument('--weights',type=str,help="Set of weights to initialize the model with")
    parser.add_argument('-p2','--prefix2',type=str,required=False,help="Second prefix for training/test files for combined training: <prefix>[train|test][num].types")
    parser.add_argument('-d2','--data_root2',type=str,required=False,help="Root folder for relative paths in second train/test files for combined training",default='')
    parser.add_argument('--data_ratio',type=float,required=False,help="Ratio to combine training data from 2 sources",default=None)
    parser.add_argument('--test_only',action='store_true',default=False,help="Don't train, just evaluate test nets once")
    parser.add_argument('--clip_gradients',type=float,default=10.0,help="Clip gradients threshold (default 10)")
    args = parser.parse_args(argv)
    
    argdict = vars(args)
    line = ''
    for (name,val) in argdict.items():
        if val != parser.get_default(name):
            line += ' --%s=%s' %(name,val)

    return (args,line)


def check_file_exists(file):
    if not os.path.isfile(file):
        raise OSError('%s does not exist' % file)


def get_train_test_files(prefix, foldnums, allfolds, reduced, prefix2):
    files = {}
    if foldnums is None:
        foldnums = set()
        glob_files = glob.glob(prefix + '*')
        if prefix2:
            glob_files += glob.glob(prefix2 + '*')
        pattern = r'(%s|%s)(reduced)?(train|test)(\d+)\.types$' % (prefix, prefix2)
        for file in glob_files:
            match = re.match(pattern, file)
            if match:
                foldnums.add(int(match.group(4)))
    elif isinstance(foldnums, str):
        foldnums = [int(i) for i in foldnums.split(',') if i]
    for i in foldnums:
        files[i] = {}
        files[i]['train'] = '%strain%d.types' % (prefix, i)
        files[i]['test'] = '%stest%d.types' % (prefix, i)
        if reduced:
            files[i]['reduced_train'] = '%sreducedtrain%d.types' % (prefix, i)
            files[i]['reduced_test'] = '%sreducedtest%d.types' % (prefix, i)
        if prefix2:
            files[i]['train2'] = '%strain%d.types' % (prefix2, i)
            files[i]['test2'] = '%stest%d.types' % (prefix2, i)
            if reduced:
                files[i]['reduced_train2'] = '%sreducedtrain%d.types' % (prefix2, i)
                files[i]['reduced_test2'] = '%sreducedtest%d.types' % (prefix2, i)
    if allfolds:
        i = 'all'
        files[i] = {}
        files[i]['train'] = files[i]['test'] = '%s.types' % prefix
        if reduced:
            files[i]['reduced_train'] = files[i]['reduced_test'] = '%sreduced.types' % prefix
        if prefix2:
            files[i]['train2'] = files[i]['test2'] = '%s.types' % prefix2
            if reduced:
                files[i]['reduced_train2'] = files[i]['reduced_test2'] = '%sreduced.types' % prefix2
    for i in files:
        for file in files[i].values():
            check_file_exists(file)
    return files


if __name__ == '__main__':
    (args,cmdline) = parse_args()

    #identify all train/test pairs
    try:
        train_test_files = get_train_test_files(args.prefix, args.foldnums, args.allfolds, args.reduced, args.prefix2)
    except OSError as e:
        print "error: %s" % e
        sys.exit(1)

    if len(train_test_files) == 0:
        print "error: missing train/test files"
        sys.exit(1)

    for i in train_test_files:
        for key in sorted(train_test_files[i], key=len):
            print str(i).rjust(3), key.rjust(14), train_test_files[i][key]

    outprefix = args.outprefix
    if outprefix == '':
        outprefix = '%s.%d' % (os.path.splitext(os.path.basename(args.model))[0],os.getpid())
        args.outprefix = outprefix

    test_aucs, train_aucs = [], []
    test_rmsds, train_rmsds = [], []
    test_y_true, train_y_true = [], []
    test_y_score, train_y_score = [], []
    test_y_aff, train_y_aff = [], []
    test_y_predaff, train_y_predaff = [], []
    test_rmsd_rmses,train_rmsd_rmses = [], []
    test_rmsd_pred, train_rmsd_pred = [], []
    test_rmsd_true, train_rmsd_true = [], []
    test2_aucs, train2_aucs = [], []
    test2_rmsds, train2_rmsds = [], []
    test2_y_true, train2_y_true = [], []
    test2_y_score, train2_y_score = [], []
    test2_y_aff, train2_y_aff = [], []
    test2_y_predaff, train2_y_predaff = [], []

    checkfold = -1
    if args.checkpoint:
        #check for existence of checkpoint
        cmdcheckname = '%s.cmdline.CHECKPOINT'%outprefix
        if os.path.exists(cmdcheckname):
            #validate this is the same
            #figure out where we were
            oldline = open(cmdcheckname).read()
            if oldline != cmdline:
                print oldline
                print "Previous commandline from checkpoint does not match current.  Cannot restore checkpoint."
                sys.exit(1)
        
        outcheck = open(cmdcheckname,'w')
        outcheck.write(cmdline)
        outcheck.close()        
        
    #train each pair
    numfolds = 0
    for i in train_test_files:

        outname = '%s.%s' % (outprefix, i)        
        cont = args.cont
                
        results = train_and_test_model(args, train_test_files[i], outname, cont)

        if args.prefix2:
            test, train, test2, train2 = results
        else:
            test, train = results

        #write out the final predictions for test and train sets
        if test.aucs:
            write_results_file('%s.auc.finaltest' % outname, test.y_true, test.y_score, footer='AUC %f\n' % test.aucs[-1])
            write_results_file('%s.auc.finaltrain' % outname, train.y_true, train.y_score, footer='AUC %f\n' % train.aucs[-1])

        if test.rmsds:
            write_results_file('%s.rmsd.finaltest' % outname, test.y_aff, test.y_predaff, footer='RMSD %f\n' % test.rmsds[-1])
            write_results_file('%s.rmsd.finaltrain' % outname, train.y_aff, train.y_predaff, footer='RMSD %f\n' % train.rmsds[-1])

        if test.rmsd_rmses:
            write_results_file('%s.rmsd_rmse.finaltest' % outname, test.rmsd_true, test.rmsd_pred, footer='RMSE %f\n' % test.rmsd_rmses[-1])
            write_results_file('%s.rmsd_rmse.finaltrain' % outname, train.rmsd_true, train.rmsd_pred, footer='RMSE %f\n' % train.rmsd_rmses[-1])

        if args.prefix2:
            if test2.aucs:
                write_results_file('%s.auc.finaltest2' % outname, test2.y_true, test2.y_score, footer='AUC %f\n' % test2.aucs[-1])
                write_results_file('%s.auc.finaltrain2' % outname, train2.y_true2, train2.y_score, footer='AUC %f\n' % train2.aucs[-1])

            if test2.rmsds:
                write_results_file('%s.rmsd.finaltest2' % outname, test2.y_aff, test2.y_predaff, footer='RMSD %f\n' % test2.rmsds[-1])
                write_results_file('%s.rmsd.finaltrain2' % outname, train2.y_aff, train2.y_predaff, footer='RMSD %f\n' % train2.rmsds[-1])

        if i == 'all':
            continue
        numfolds += 1

        #aggregate results from different crossval folds
        if test.aucs:
            test_aucs.append(test.aucs)
            train_aucs.append(train.aucs)
            test_y_true.extend(test.y_true)
            test_y_score.extend(test.y_score)
            train_y_true.extend(train.y_true)
            train_y_score.extend(train.y_score)

        if test.rmsds:
            test_rmsds.append(test.rmsds)
            train_rmsds.append(train.rmsds)
            test_y_aff.extend(test.y_aff)
            test_y_predaff.extend(test.y_predaff)
            train_y_aff.extend(train.y_aff)
            train_y_predaff.extend(train.y_predaff)
            
        if test.rmsd_rmses:
            test_rmsd_rmses.append(test.rmsd_rmses)
            train_rmsd_rmses.append(train.rmsd_rmses)
            test_rmsd_true.extend(test.rmsd_true)
            test_rmsd_pred.extend(test.rmsd_pred)
            train_rmsd_true.extend(train.rmsd_true)
            train_rmsd_pred.extend(train.rmsd_pred)            

        if args.prefix2:
            if test2.aucs:
                test2_aucs.append(test2.aucs)
                train2_aucs.append(train2.aucs)
                test2_y_true.extend(test2.y_true)
                test2_y_score.extend(test2.y_score)
                train2_y_true.extend(train2.y_true)
                train2_y_score.extend(train2.y_score)

            if test2.rmsds:
                test2_rmsds.append(test2.rmsds)
                train2_rmsds.append(train2.rmsds)
                test2_y_aff.extend(test2.y_aff)
                test2_y_predaff.extend(test2.y_predaff)
                train2_y_aff.extend(train2.y_aff)
                train2_y_predaff.extend(train2.y_predaff)

    #only combine fold results if we have multiple folds
    if numfolds > 1:

        if any(test_aucs):
            combine_fold_results(test_aucs, train_aucs, test_y_true, test_y_score, train_y_true, train_y_score,
                                 outprefix, args.test_interval, affinity=False, second_data_source=False)

        if any(test_rmsds):
            combine_fold_results(test_rmsds, train_rmsds, test_y_aff, test_y_predaff, train_y_aff, train_y_predaff,
                                 outprefix, args.test_interval, affinity=True, second_data_source=False,
                                 filter_actives_test=test_y_true, filter_actives_train=train_y_true)

        if any(test_rmsds):
            combine_fold_results(test_rmsds, train_rmsds, test_y_aff, test_y_predaff, train_y_aff, train_y_predaff,
                                 outprefix, args.test_interval, affinity=True, second_data_source=False,
                                 filter_actives_test=test_y_true, filter_actives_train=train_y_true)


        if any(test_rmsd_rmses):
            combine_fold_results(test_rmsd_rmses, train_rmsd_rmses, test_rmsd_true, test_rmsd_pred, train_rmsd_true, train_rmsd_pred,
                                 outprefix, args.test_interval, affinity=True, second_data_source=False)
                                 
                                 
        if any(test2_aucs):
            combine_fold_results(test2_aucs, train2_aucs, test2_y_true, test2_y_score, train2_y_true, train2_y_score,
                                 outprefix, args.test_interval, affinity=False, second_data_source=True)

        if any(test2_rmsds):
            combine_fold_results(test2_rmsds, train2_rmsds, test2_y_aff, test2_y_predaff, train2_y_aff, train2_y_predaff,
                                 outprefix, args.test_interval, affinity=True, second_data_source=True,
                                 filter_actives_test=test2_y_true, filter_actives_train=train2_y_true)

