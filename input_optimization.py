#!/usr/bin/env python

import caffe
from train import write_model_file, check_file_exists, count_lines
from combine_fold_results import write_results_file
import argparse, glob, re, os, itertools
import numpy as np
from caffe.proto.caffe_pb2 import NetParameter
import google.protobuf.text_format as prototxt
import sklearn.metrics

'''
Use trained network to optimize grids associated with input structures. Need a
model template, data, and pre-trained weights. 
'''

#layers that don't need an lr_mult because they don't have trainable parameters
no_param_layers = ['Pooling', 'ELU', 'ReLU', 'PReLU', 'Sigmoid', 'TanH', 'Power', 'Exp',
        'Log', 'BNLL', 'Threshold', 'Bias', 'Scale', 'Softmax', 'AffinityLoss', 
        'SoftmaxWithLoss', 'Reshape', 'Split']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize gnina input grid a"
    " la DeepDream")
    parser.add_argument('-m', '--model', type=str, required=True, help="Model"
    " template file defining network architecture")
    parser.add_argument('-i', '--iterations', type=int, required=False, 
            help="Number of iterations to run, default 100", default=100)
    parser.add_argument('-d', '--data_root', type=str, required=False, 
            help="Root folder for relative paths in train/test files", default='')
    parser.add_argument('-p', '--prefix', type=str, required=False,
            help="Prefix for file(s) containing input data", default='')
    parser.add_argument('--weights', type=str, help="Set of weights to"
            " initialize the model with", required=True)
    parser.add_argument('-t', '--trained_net', type=str, help="Net associated"
            " with trained weights", required=True)
    parser.add_argument('-er', '--exclude_receptor', default=False,
            action='store_true', help='Only update the ligand')
    parser.add_argument('-el', '--exclude_ligand', default=False,
            action='store_true', help='Only update the receptor')
    parser.add_argument('-o', '--outprefix', type=str, help="Prefix for output files,"
    " default dream_<model>.<pid>", default='')
    parser.add_argument('-g', '--gpu',type=int, help='Specify GPU to run on', default=-1)
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate"
            " for input grid updates, default=0.01.")
    parser.add_argument('--threshold', type=float, default=1e-5,
            help="Convergence threshold for early termination, default 1e-5")
    parser.add_argument('-an', '--allow_negative', default=False,
            action='store_true', help="Allow negative density, useful if the"
            "result is to be used for a similarity search rather than to"
            "represent a physical molecule")

args = parser.parse_args()
assert not (args.exclude_receptor and args.exclude_ligand), "Must optimize at least one of receptor and ligand"
pid = os.getpid()

if args.gpu >= 0:
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()

glob_files = glob.glob(args.prefix + '*')
train_files = []
pattern = r'(%s)(reduced)?(train|test)(\d+)\.types$' % (args.prefix)
for f in glob_files:
    match = re.match(pattern, f)
    if match:
    #if globbing with the prefix returns a train/test pair, just take the test fold
        if match.group(3) == 'test':
            train_files.append(f)
    else:
    #if it doesn't match the pattern, assume the user really meant to use that exact file
        train_files.append(f)
if not len(train_files):
    raise OSError("Prefix matches no files")

#check that in the template net:
#lr_mult==0 for non-data layers 
#dream==True for molgrid layer
#also retrieve recmap and ligmap
with open(args.model) as f:
    netparam = NetParameter()
    prototxt.Merge(f.read(), netparam)
    for layer in netparam.layer:
        if layer.type == "MolGridData":
            assert layer.molgrid_data_param.dream, "molgrid layer must have dream: True"
            ligmap = layer.molgrid_data_param.ligmap
            recmap = layer.molgrid_data_param.recmap
        elif layer.type not in no_param_layers:
            assert len(layer.param)==2, "must set lr_mult for weights and bias"
            for i in range(2):
                assert layer.param[i].lr_mult==0.0, "lr_mult for non-data layers must be 0"
                assert layer.param[i].decay_mult==0.0, "decay mult for non-data layers must be 0"

#figure out which channels to exclude from updates if applicable
#rec channels always come first
if recmap:
    nrec_channels = count_lines(recmap)
else:
    nrec_channels = 16
if ligmap:
    nlig_channels = count_lines(ligmap)
else:
    nlig_channels = 19

for train_file in train_files:
    base = os.path.splitext(os.path.basename(train_file))[0]
    if not args.outprefix:
        outname = 'dream_%s.%d' %(base, pid)
    else:
        if len(train_files) > 1:
            outname = '%s_%s' %(args.outprefix, base)
        else:
            outname = args.outprefix
    test_model = '%s.prototxt' % outname
    write_model_file(test_model, args.model, train_file, train_file, args.data_root, True)

    #construct net and update weights
    check_file_exists(args.weights)
    check_file_exists(args.trained_net)
    trained_net = caffe.Net(args.trained_net, caffe.TRAIN,
            weights=args.weights)
    dream_net = caffe.Net(test_model, caffe.TRAIN)
    for layer in trained_net.params:
        if layer in dream_net.params:
            dream_net.params[layer][0].data[...] = trained_net.params[layer][0].data[...]
            dream_net.params[layer][1].data[...] = trained_net.params[layer][1].data[...]
   
    diffs = []
    #do forward, backward, update input grid for desired number of iters 
    #(molgrid currently handles grid dumping)
    #TODO: momentum?
    all_y_scores = []
    for i in xrange(args.iterations):
        res = dream_net.forward()
        assert 'output' in res, "Network must produce output"
        all_y_scores.append([float(x[1]) for x in res['output']])
        dream_net.backward()
        current_grid = dream_net.blobs['data'].data
        grid_diff = dream_net.blobs['data'].diff
        #TODO: assumes second dim is atom type channel
        norm = np.linalg.norm(grid_diff)
        diffs.append(norm)
        if norm < args.threshold:
            print "gradient sufficiently converged, terminating early\n"
            break
        assert len(current_grid.shape) == 5, "Currently require grids to be NxCxDIMxDIMxDIM"
        assert len(grid_diff.shape) == 5, "Currently require grids to be NxCxDIMxDIMxDIM"
        if args.exclude_receptor:
            current_grid[:,nrec_channels:,:,:,:] -= args.lr * grid_diff[:,nrec_channels:,:,:,:]
        elif args.exclude_ligand:
            current_grid[:,:nrec_channels,:,:,:] -= args.lr * grid_diff[:,nrec_channels:,:,:,:]
        else:
            current_grid -= args.lr * grid_diff
        #don't let anything go negative, if desired
        if not args.allow_negative:
            current_grid[current_grid < 0] = 0
        dream_net.blobs['data'].data[...] = current_grid
    #do final evaluation, write to output files
    res = dream_net.forward()
    y_true = []
    y_score = []
    y_affinity = []
    y_predaff = []
    losses = []

    all_y_scores = [list(i) for i in zip(*all_y_scores)]
    write_results_file('%s.diffs' % outname, diffs)
    write_results_file('%s.preds' % outname, *all_y_scores)

    if 'labelout' in res:
        y_true = [float(x) for x in res['labelout']]

    if 'output' in res:
        y_score = [float(x[1]) for x in res['output']]

    if 'affout' in res:
        y_affinity = [float(x) for x in res['affout']]

    if 'predaff' in res:
        y_predaff = [float(x) for x in res['predaff']]

    if 'loss' in res:
        print "%s loss: %f\n" %(base, res['loss'])

    #compute auc
    if y_true and y_score:
        if len(np.unique(y_true)) > 1:
            auc = sklearn.metrics.roc_auc_score(y_true, y_score)
        else: # may be evaluating all crystal poses?
            print "Warning: only one unique label"
            auc = 1.0
        write_results_file('%s.auc.finaltest' % outname, y_true, y_score,
                footer='AUC %f\n' % auc)

    #compute mean squared error (rmsd) of affinity (for actives only)
    if y_affinity and y_predaff:
        y_predaff_true = np.array(y_predaff)[np.array(y_affinity)>0]
        y_aff_true = np.array(y_affinity)[np.array(y_affinity)>0]
        rmsd = np.sqrt(sklearn.metrics.mean_squared_error(y_aff_true, y_predaff_true))
        write_results_file('%s.rmsd.finaltest' % outname, y_affinity,
                y_predaff, footer='RMSD %f\n' % rmsd)
