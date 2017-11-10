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
import psutil
from combine_fold_results import write_results_file, combine_fold_results, filter_actives


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
                      lr_policy, gamma, power, random_seed, max_iter, snapshot_prefix):
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
    param.snapshot_prefix = snapshot_prefix
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
    y_true = np.zeros(n_tests)
    y_score = np.zeros((n_tests, n_rotations))
    y_loss = np.zeros((n_tests, n_rotations))
    aff_true = np.zeros(n_tests)
    aff_pred = np.zeros((n_tests, n_rotations))
    aff_loss = np.zeros((n_tests, n_rotations))
    rmsd_true = np.zeros(n_tests)
    rmsd_pred = np.zeros((n_tests, n_rotations))
    rmsd_loss = np.zeros((n_tests, n_rotations))

    #get relevant test net blobs, or None if they don't exist
    y_true_blob = test_net.blobs.get('label')
    y_score_blob = test_net.blobs.get('output')
    y_loss_blob = test_net.blobs.get('loss')
    aff_true_blob = test_net.blobs.get('affinity')
    aff_pred_blob = test_net.blobs.get('predaff')
    aff_loss_blob = test_net.blobs.get('aff_loss')
    rmsd_true_blob = test_net.blobs.get('rmsd_true')
    rmsd_pred_blob = test_net.blobs.get('rmsd_pred')
    rmsd_loss_blob = test_net.blobs.get('rmsd_loss')

    batch_size = i = test_net.blobs['data'].shape[0]
    for r in xrange(n_rotations):
        for x in xrange(n_tests):
            x = (x + offset) % n_tests

            #evaluate next batch as needed
            if i >= batch_size:
                test_net.forward()
                i = 0

            if y_true_blob:
                if r == 0:
                    y_true[x] = float(y_true_blob.data[i])
                else:
                    assert y_true[x] == y_true_blob.data[i] #sanity check

            if y_score_blob:
                if y_score_blob.shape[1] == 2:
                    y_score[x][r] = float(y_score_blob.data[i][1])
                else:
                    y_score[x][r] = float(y_score_blob.data[i][0])

            if y_loss_blob:
                y_loss[x][r] = float(y_loss_blob.data)

            if aff_true_blob:
                if r == 0:
                    aff_true[x] = float(aff_true_blob.data[i])
                else:
                    assert aff_true[x] == aff_true_blob.data[i] #sanity check

            if aff_pred_blob:
                aff_pred[x][r] = float(aff_pred_blob.data[i])

            if aff_loss_blob:
                aff_loss[x][r] = float(aff_loss_blob.data)

            if rmsd_true_blob:
                if r == 0:
                    rmsd_true[x] = float(rmsd_true_blob.data[i])
                else:
                    assert rmsd_true[x] == rmsd_true_blob.data[i] #sanity check

            if rmsd_pred_blob:
                rmsd_pred[x][r] = float(rmsd_pred_blob.data[i])

            if rmsd_loss_blob:
                rmsd_loss[x][r] = float(rmsd_loss_blob.data)

            i += 1

    #get index of test example that will be at start of next batch
    offset = (x + 1 + batch_size - i) % n_tests

    #average predictions across each rotation
    y_score = y_score.mean(axis=1)
    aff_pred = aff_pred.mean(axis=1)
    rmsd_pred = rmsd_pred.mean(axis=1)

    #average loss across entire test set and rotations
    y_loss = y_loss.mean()
    aff_loss = aff_loss.mean()
    rmsd_loss = rmsd_loss.mean()

    #compute auc
    if y_true_blob and y_score_blob:
        if len(np.unique(y_true)) > 1:
            auc = sklearn.metrics.roc_auc_score(y_true, y_score)
        else: # may be evaluating all crystal poses?
            print "Warning: only one unique label"
            auc = -1.0
    else:
        auc = None

    #compute root mean squared error (rmse) of affinity
    if aff_true_blob and aff_pred_blob:
        #remove negative affinities
        aff_true_pos = aff_true[aff_true >= 0]
        aff_pred_pos = aff_pred[aff_true >= 0]
        #if there are labels, remove decoys
        if y_true_blob:
            aff_pred_active = filter_actives(aff_pred_pos, y_true)
            aff_true_active = filter_actives(aff_true_pos, y_true)
            aff_rmse = np.sqrt(sklearn.metrics.mean_squared_error(aff_pred_active, aff_true_active))
        else:
            aff_rmse = np.sqrt(sklearn.metrics.mean_squared_error(aff_pred_pos, aff_true_pos))
    else:
        aff_rmse = None

    #compute root mean squared error (rmse) of rmsd
    if rmsd_true_blob and rmsd_pred_blob:
        rmsd_rmse = np.sqrt(sklearn.metrics.mean_squared_error(rmsd_pred, rmsd_true))
    else:
        rmsd_rmse = None

    #put test results in a namespace object to better organize different test_nets
    result = Namespace()
    result.y_true = y_true
    result.y_score = y_score
    result.y_loss = y_loss
    result.auc = auc
    result.aff_true = aff_true
    result.aff_pred = aff_pred
    result.aff_loss = aff_loss
    result.aff_rmse = aff_rmse
    result.rmsd_true = rmsd_true
    result.rmsd_pred = rmsd_pred
    result.rmsd_loss = rmsd_loss
    result.rmsd_rmse = rmsd_rmse
    return result, offset


def append_test_result(name, results, new_result, check_alignment):

    if check_alignment:
        assert np.all(new_result.y_true == results.y_true)
        assert np.all(new_result.aff_true == results.aff_true)
        assert np.all(new_result.rmsd_true == results.rmsd_true)

    results.y_true = new_result.y_true
    results.y_score = new_result.y_score
    results.aff_true = new_result.aff_true
    results.aff_pred = new_result.aff_pred
    results.rmsd_true = new_result.rmsd_true
    results.rmsd_pred = new_result.rmsd_pred

    if new_result.auc is not None:
        print "%s auc = %f" % (name, new_result.auc)
        results.auc.append(new_result.auc)

        if new_result.y_loss is not None:
            print "%s y_loss = %f" % (name, new_result.y_loss)
            results.y_loss.append(new_result.y_loss)

    if new_result.aff_rmse is not None:
        print "%s aff_rmse = %f" % (name, new_result.aff_rmse)
        results.aff_rmse.append(new_result.aff_rmse)

        if new_result.aff_loss is not None:
            print "%s aff_loss = %f" % (name, new_result.aff_loss)
            results.aff_loss.append(new_result.aff_loss)

    if new_result.rmsd_rmse is not None:
        print "%s rmsd_rmse = %f" % (name, new_result.rmsd_rmse)
        results.rmsd_rmse.append(new_result.rmsd_rmse)

        if new_result.rmsd_loss is not None:
            print "%s rmsd_loss = %f" % (name, new_result.rmsd_loss)
            results.rmsd_loss.append(new_result.rmsd_loss)


def get_metric_names(results, prefix):
    names = []
    if results.auc:
        names.append(prefix + '_auc')
    if results.y_loss:
        names.append(prefix + '_y_loss')
    if results.aff_rmse:
        names.append(prefix + '_aff_rmse')
    if results.aff_loss:
        names.append(prefix + '_aff_loss')
    if results.rmsd_rmse:
        names.append(prefix + '_rmsd_rmse')
    if results.rmsd_loss:
        names.append(prefix + '_rmsd_loss')
    return names


def get_last_test_metrics(results):
    metrics = []
    if results.auc:
        metrics.append(results.auc[-1])
    if results.y_loss:
        metrics.append(results.y_loss[-1])
    if results.aff_rmse:
        metrics.append(results.aff_rmse[-1])
    if results.aff_loss:
        metrics.append(results.aff_loss[-1])
    if results.rmsd_rmse:
        metrics.append(results.rmsd_rmse[-1])
    if results.rmsd_loss:
        metrics.append(results.rmsd_loss[-1])
    return metrics


def count_lines(file):
    return sum(1 for line in open(file, 'r'))


def train_and_test_model(args, files, outname):
    '''Train caffe model for iterations steps using provided model template
    and training file(s), and every test_interval iterations evaluate each
    of the train and test files. Return AUC (and RMSE, if affinity model)
    for every test iteration, and also the labels and predictions for the
    final test iteration.'''
    template = args.model
    test_interval = args.test_interval
    iterations = args.iterations
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

    #write solver prototxt
    solverf = 'solver.%d.prototxt' % pid
    write_solver_file(solverf, test_models[0], test_models, args.solver, args.base_lr, args.momentum, args.weight_decay,
                      args.lr_policy, args.gamma, args.power, args.seed, iterations+args.cont, outname)

    #set up solver in caffe
    if args.gpu >= 0:
        caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solverf)
    if args.cont:
        modelname = '%s_iter_%d.caffemodel' % (outname, args.cont)
        solvername = '%s_iter_%d.solverstate' % (outname, args.cont)
        check_file_exists(solvername)
        solver.restore(solvername)
        solver.testall() #link testnets to train net
    if args.weights:
        check_file_exists(args.weights)
        solver.net.copy_from(args.weights) #TODO this doesn't actually set the necessary weights...

    test_nets = {}
    for key, test_file in files.items():
        idx = test_files.index(test_file)
        test_nets[key] = solver.test_nets[idx], count_lines(test_file), 0

    if training: #outfile is training progress, don't write if we're not training
        if args.cont: #TODO changes in test_interval not reflected in outfile
            mode = 'a'
        else:
            mode = 'w'
        outfile = '%s.out' % outname
        out = open(outfile, mode, 0) #unbuffered

    #store test evaluation metrics, test labels and most recent predictions in namespace objects
    train = Namespace(auc=[],y_loss=[],aff_rmse=[],aff_loss=[],rmsd_rmse=[],rmsd_loss=[])
    if test_on_train:
        test = train
    else:
        test = Namespace(auc=[],y_loss=[],aff_rmse=[],aff_loss=[],rmsd_rmse=[],rmsd_loss=[])
    if args.prefix2:
        train2 = Namespace(auc=[],y_loss=[],aff_rmse=[],aff_loss=[],rmsd_rmse=[],rmsd_loss=[])
        if test_on_train:
            test2 = train2
        else:
            test2 = Namespace(auc=[],y_loss=[],aff_rmse=[],aff_loss=[],rmsd_rmse=[],rmsd_loss=[])

    #also keep track of best test and train aucs
    best_test_auc = 0
    best_train_auc = 0
    best_test_aff_rmse = np.inf
    best_test_rmsd_rmse = np.inf
    best_train_interval = 0

    i_time_avg = 0
    for i in xrange(iterations/test_interval):
        last_test = i == iterations/test_interval-1
        check_alignment = i > 0 and not (args.reduced and last_test)

        i_start = start = time.time()
        if training:
            #train
            solver.step(test_interval)
            print "Iteration %d" % (args.cont + (i+1)*test_interval)
            print "Train time: %f" % (time.time()-start)

        if not test_on_train:
            #evaluate test set
            start = time.time()
            if args.reduced and not last_test:
                key = 'reduced_test'
            else:
                key = 'test'
            test_net, n_tests, offset = test_nets[key]
            result, offset = evaluate_test_net(test_net, n_tests, rotations, offset)
            test_nets[key] = test_net, n_tests, offset
            append_test_result('test', test, result, check_alignment)
            print "Eval test time: %f" % (time.time()-start)

            if args.prefix2:
                #evaluate test set 2
                start = time.time()
                if args.reduced and not last_test:
                    key = 'reduced_test2'
                else:
                    key = 'test2'
                test_net, n_tests, offset = test_nets[key]
                result, offset = evaluate_test_net(test_net, n_tests, rotations, offset)
                test_nets[key] = test_net, n_tests, offset
                append_test_result('test2', test2, result, check_alignment)
                print "Eval test2 time: %f" % (time.time()-start)

        #evaluate train set
        start = time.time()
        if args.reduced and not last_test:
            key = 'reduced_train'
        else:
            key = 'train'
        test_net, n_tests, offset = test_nets[key]
        result, offset = evaluate_test_net(test_net, n_tests, rotations, offset)
        test_nets[key] = test_net, n_tests, offset
        append_test_result('train', train, result, check_alignment)
        print "Eval train time: %f" % (time.time()-start)

        if args.prefix2:
            #evaluate train set 2
            start = time.time()
            if args.reduced and not last_test:
                key = 'reduced_train2'
            else:
                key = 'train2'
            test_net, n_tests, offset = test_nets[key]
            result, offset = evaluate_test_net(test_net, n_tests, rotations, offset)
            test_nets[key] = test_net, n_tests, offset
            append_test_result('train2', train2, result, check_alignment)
            print "Eval train2 time: %f" % (time.time()-start)

        if training:
            #check for auc improvement
            if test.auc:
                if test.auc[-1] > best_test_auc:
                    best_test_auc = test.auc[-1]
                    if args.keep_best:
                        solver.snapshot() #a bit too much - gigabytes of data
                if train.auc[-1] > best_train_auc:
                    best_train_auc = train.auc[-1]
                    best_train_auc_i = i
                if args.dynamic:
                    lr = solver.get_base_lr()
                    if (i - best_train_interval) > args.step_when: #reduce learning rate
                        lr *= args.step_reduce
                        solver.set_base_lr(lr)
                        best_train_interval = i #reset
                        best_train_auc = train.auc[-1] #the value too, so we can consider the recovery
                    if lr < args.step_end:
                        break #end early
            #check for aff_rmse improvement
            if test.aff_rmse:
                if test.aff_rmse[-1] < best_test_aff_rmse:
                    best_test_aff_rmse = test.aff_rmse[-1]
                    if args.keep_best:
                        solver.snapshot() #a bit too much - gigabytes of data
            #check for rmsd_rmse improvement
            if test.rmsd_rmse:
                if test.rmsd_rmse[-1] < best_test_rmsd_rmse:
                    best_test_rmsd_rmse = test.rmsd_rmse[-1]
                    if args.keep_best:
                        solver.snapshot() #a bit too much - gigabytes of data

            #write out evaluation results
            row = get_last_test_metrics(test) + get_last_test_metrics(train)
            if args.prefix2:
                row += get_last_test_metrics(test2) + get_last_test_metrics(train2)
            row.append(solver.get_base_lr())
            if i == 0:
                col_names = get_metric_names(test, 'test') + get_metric_names(train, 'train')
                if args.prefix2:
                    col_names += get_metric_names(test2, 'test2') + get_metric_names(train2, 'train2')
                col_names.append('base_lr')
                out.write(' '.join(col_names) + '\n')
            out.write(' '.join('%.6f' % x for x in row) + '\n')
            out.flush()

        #track avg time per loop
        i_time = time.time()-i_start
        i_time_avg = (i*i_time_avg + i_time)/(i+1)
        i_left = iterations/test_interval - (i+1)
        time_left = i_time_avg * i_left
        time_str = time.strftime('%H:%M:%S', time.gmtime(time_left))
        print "Loop time: %f (%s left)" % (i_time, time_str)

        mem = psutil.Process(os.getpid()).memory_info().rss
        print "Memory usage: %.3fgb (%d)" % (mem/1073741824., mem)

    if training:
        out.close()
        solver.snapshot()
    del solver #free mem

    if not args.keep:
        os.remove(solverf)
        for test_model in test_models:
            os.remove(test_model)

    if args.prefix2:
        return test, train, test2, train2
    else:
        return test, train


def parse_args(argv=None):
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
    parser.add_argument('-p2','--prefix2',type=str,required=False,help="Second prefix for training/test files for combined training: <prefix>[train|test][num].types")
    parser.add_argument('-d2','--data_root2',type=str,required=False,help="Root folder for relative paths in second train/test files for combined training",default='')
    parser.add_argument('--data_ratio',type=float,required=False,help="Ratio to combine training data from 2 sources",default=None)
    parser.add_argument('--test_only',action='store_true',default=False,help="Don't train, just evaluate test nets once")
    return parser.parse_args(argv)


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
        pattern = r'(%s|%s)(_reduced)?(train|test)(\d+)\.types$' % (prefix, prefix2)
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
            files[i]['reduced_train'] = '%s_reducedtrain%d.types' % (prefix, i)
            files[i]['reduced_test'] = '%s_reducedtest%d.types' % (prefix, i)
        if prefix2:
            files[i]['train2'] = '%strain%d.types' % (prefix2, i)
            files[i]['test2'] = '%stest%d.types' % (prefix2, i)
            if reduced:
                files[i]['reduced_train2'] = '%s_reducedtrain%d.types' % (prefix2, i)
                files[i]['reduced_test2'] = '%s_reducedtest%d.types' % (prefix2, i)
    if allfolds:
        i = 'all'
        files[i] = {}
        files[i]['train'] = files[i]['test'] = '%s.types' % prefix
        if reduced:
            files[i]['reduced_train'] = files[i]['reduced_test'] = '%s_reduced.types' % prefix
        if prefix2:
            files[i]['train2'] = files[i]['test2'] = '%s.types' % prefix2
            if reduced:
                files[i]['reduced_train2'] = files[i]['reduced_test2'] = '%s_reduced.types' % prefix2
    for i in files:
        for file in files[i].values():
            check_file_exists(file)
    return files


if __name__ == '__main__':
    args = parse_args()

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

    test_auc, train_auc = [], []
    test_y_true, train_y_true = [], []
    test_y_score, train_y_score = [], []

    test_aff_rmse, train_aff_rmse = [], []
    test_aff_true, train_aff_true = [], []
    test_aff_pred, train_aff_pred = [], []

    test_rmsd_rmse, train_rmsd_rmse = [], []
    test_rmsd_true, train_rmsd_true = [], []
    test_rmsd_pred, train_rmsd_pred = [], []

    test2_auc, train2_auc = [], []
    test2_y_true, train2_y_true = [], []
    test2_y_score, train2_y_score = [], []

    test2_aff_rmse, train2_aff_rmse = [], []
    test2_aff_true, train2_aff_true = [], []
    test2_aff_pred, train2_aff_pred = [], []

    test2_rmsd_rmse, train2_rmsd_rmse = [], []
    test2_rmsd_true, train2_rmsd_true = [], []
    test2_rmsd_pred, train2_rmsd_pred = [], []

    #train each pair
    numfolds = 0
    for i in train_test_files:

        outname = '%s.%s' % (outprefix, i)
        results = train_and_test_model(args, train_test_files[i], outname)

        if args.prefix2:
            test, train, test2, train2 = results
        else:
            test, train = results

        #write out the final predictions for test and train sets
        if test.auc:
            write_results_file('%s.auc.finaltest' % outname,
                test.y_true, test.y_score, footer='auc = %f\n' % test.auc[-1])
            write_results_file('%s.auc.finaltrain' % outname,
                train.y_true, train.y_score, footer='auc = %f\n' % train.auc[-1])

        if test.aff_rmse:
            write_results_file('%s.aff_rmse.finaltest' % outname,
                test.aff_true, test.aff_pred, footer='aff_rmse = %f\n' % test.aff_rmse[-1])
            write_results_file('%s.aff_rmse.finaltrain' % outname,
                train.aff_true, train.aff_pred, footer='aff_rmse = %f\n' % train.aff_rmse[-1])

        if test.rmsd_rmse:
            write_results_file('%s.rmsd_rmse.finaltest' % outname,
                test.rmsd_true, test.rmsd_pred, footer='rmsd_rmse = %f\n' % test.rmsd_rmse[-1])
            write_results_file('%s.rmsd_rmse.finaltrain' % outname,
                train.rmsd_true, train.rmsd_pred, footer='rmsd_rmse = %f\n' % train.rmsd_rmse[-1])

        if args.prefix2:
            if test2.auc:
                write_results_file('%s.auc.finaltest2' % outname,
                    test2.y_true, test2.y_score, footer='auc = %f\n' % test2.auc[-1])
                write_results_file('%s.auc.finaltrain2' % outname,
                    train2.y_true, train2.y_score, footer='auc = %f\n' % train2.auc[-1])

            if test2.aff_rmse:
                write_results_file('%s.aff_rmse.finaltest2' % outname,
                    test2.aff_true, test2.aff_pred, footer='aff_rmse = %f\n' % test2.aff_rmse[-1])
                write_results_file('%s.aff_rmse.finaltrain2' % outname,
                    train2.aff_true, train2.aff_pred, footer='aff_rmse = %f\n' % train2.aff_rmse[-1])

            if test2.rmsd_rmse:
                write_results_file('%s.rmsd_rmse.finaltest2' % outname,
                    test2.rmsd_true, test2.rmsd_pred, footer='rmsd_rmse = %f\n' % test2.rmsd_rmse[-1])
                write_results_file('%s.rmsd_rmse.finaltrain2' % outname,
                    train2.rmsd_true, train2.rmsd_pred, footer='rmsd_rmse = %f\n' % train2.rmsd_rmse[-1])

        if i == 'all':
            continue
        numfolds += 1

        #aggregate results from different crossval folds
        if test.auc:
            test_auc.append(test.auc)
            train_auc.append(train.auc)
            test_y_true.extend(test.y_true)
            test_y_score.extend(test.y_score)
            train_y_true.extend(train.y_true)
            train_y_score.extend(train.y_score)

        if test.aff_rmse:
            test_aff_rmse.append(test.aff_rmse)
            train_aff_rmse.append(train.aff_rmse)
            test_aff_true.extend(test.aff_true)
            test_aff_pred.extend(test.aff_pred)
            train_aff_true.extend(train.aff_true)
            train_aff_pred.extend(train.aff_pred)

        if test.rmsd_rmse:
            test_rmsd_rmse.append(test.rmsd_rmse)
            train_rmsd_rmse.append(train.rmsd_rmse)
            test_rmsd_true.extend(test.rmsd_true)
            test_rmsd_pred.extend(test.rmsd_pred)
            train_rmsd_true.extend(train.rmsd_true)
            train_rmsd_pred.extend(train.rmsd_pred)

        if args.prefix2:
            if test2.auc:
                test2_auc.append(test2.auc)
                train2_auc.append(train2.auc)
                test2_y_true.extend(test2.y_true)
                test2_y_score.extend(test2.y_score)
                train2_y_true.extend(train2.y_true)
                train2_y_score.extend(train2.y_score)

            if test2.aff_rmse:
                test2_aff_rmse.append(test2.aff_rmse)
                train2_aff_rmse.append(train2.aff_rmse)
                test2_aff_true.extend(test2.aff_true)
                test2_aff_pred.extend(test2.aff_pred)
                train2_aff_true.extend(train2.aff_true)
                train2_aff_pred.extend(train2.aff_pred)

            if test2.rmsd_rmse:
                test2_rmsd_rmse.append(test2.rmsd_rmse)
                train2_rmsd_rmse.append(train2.rmsd_rmse)
                test2_rmsd_true.extend(test2.rmsd_true)
                test2_rmsd_pred.extend(test2.rmsd_pred)
                train2_rmsd_true.extend(train2.rmsd_true)
                train2_rmsd_pred.extend(train2.rmsd_pred)

    #only combine fold results if we have multiple folds
    if numfolds > 1:

        if any(test_auc):
            combine_fold_results(test_auc, train_auc, test_y_true, test_y_score, train_y_true, train_y_score,
                                 outprefix, args.test_interval, affinity=False, second_data_source=False)

        if any(test_aff_rmse):
            combine_fold_results(test_aff_rmse, train_aff_rmse, test_aff_true, test_aff_pred, train_aff_true, train_aff_pred,
                                 outprefix, args.test_interval, affinity=True, second_data_source=False,
                                 filter_actives_test=test_y_true, filter_actives_train=train_y_true)

        if any(test2_auc):
            combine_fold_results(test2_auc, train2_auc, test2_y_true, test2_y_score, train2_y_true, train2_y_score,
                                 outprefix, args.test_interval, affinity=False, second_data_source=True)

        if any(test2_aff_rmse):
            combine_fold_results(test2_aff_rmse, train2_aff_rmse, test2_aff_true, test2_aff_pred, train2_aff_true, train2_aff_pred,
                                 outprefix, args.test_interval, affinity=True, second_data_source=True,
                                 filter_actives_test=test2_y_true, filter_actives_train=train2_y_true)

