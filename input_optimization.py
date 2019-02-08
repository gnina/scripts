#!/usr/bin/env python

import caffe
from train import write_model_file, check_file_exists, count_lines
from combine_fold_results import write_results_file
import argparse, glob, re, os, itertools, math
import numpy as np
from caffe.proto.caffe_pb2 import NetParameter
import google.protobuf.text_format as prototxt
import sklearn.metrics

'''
Use trained network to optimize grids associated with input structures. Need a
model template, data, and pre-trained weights. 
'''

def get_channel_list(fname, prefix=''):
    '''
    Make a list of the channels being used
    '''
    channel_list = []
    with open(fname,'r') as f:
        for line in f:
            channel_list.append(prefix + '_'.join(line))
    return channel_list

def get_structure_names(fname):
    '''
    Use types file to get rec/lig names associated with each example
    '''
    names = []
    with open(fname, 'r') as f:
        for line in f:
            #strip off comments
            contents = line.split('#')[0]
            contents = contents.split()
            #then lig is last, rec next-to-last
            rec = os.path.splitext(os.path.basename(contents[-2]))[0]
            lig = os.path.splitext(os.path.basename(contents[-1]))[0]
            names.append('%s_%s' %(rec,lig))
    return names

def dump_grid_dx(outname, blob, dim, resolution, dimension, center, channels):
    '''
    For every atom type channel, dump diff in DX format to file with name
    outname_[channel]
    '''
    for chidx,channel in enumerate(channels):
        with open('%s_%s.dx' %(outname, str(channel)), 'w') as f:
            f.write('%s %d %d %d\n' %("object 1 class gridpositions counts ",
                dim, dim, dim))
            f.write('%s %.5f %.5f %.5f\n' %("origin", 
                center[0] - dimension/2.0, 
                center[1] - dimension/2.0, 
                center[2] - dimension/2.0))
            f.write('%s %.5f 0 0\n' %("delta", resolution))
            f.write('%s 0 %.5f 0\n' %("delta", resolution))
            f.write('%s 0 0 %.5f\n' %("delta", resolution))
            f.write('%s %d %d %d\n' %("object 2 class gridconnections counts",
                dim, dim, dim))
            f.write('%s %d%s\n' %("object 3 class array type double rank 0 "
                "items [ ", dim*dim*dim, "] data follows"))
            total = 0
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        f.write('{:.6e}'.format(blob[chidx][i][j][k]))
                        total += 1
                        if (total % 3 == 0):
                            f.write('\n')
                        else:
                            f.write(' ')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize gnina input grid a"
    " la DeepDream")
    parser.add_argument('-i', '--iterations', type=int, required=False, 
            help="Number of iterations to run, default 100", default=100)
    parser.add_argument('-d', '--data_root', type=str, required=False, 
            help="Root folder for relative paths in train/test files", default='')
    parser.add_argument('-p', '--prefix', type=str, required=False,
            help="Prefix for file(s) containing input data", default='')
    parser.add_argument('-w', '--weights', type=str, help="Set of weights to"
            " initialize the model with", required=True)
    parser.add_argument('-m', '--model', type=str, help="Network model associated"
            " with trained weights", required=True)
    parser.add_argument('-bs', '--batch_size', type=int, help="Batch size for \
            input optimization, defaults to value in model file",
            default=0)
    parser.add_argument('-er', '--exclude_receptor', default=False,
            action='store_true', help='Only update the ligand')
    parser.add_argument('-el', '--exclude_ligand', default=False,
            action='store_true', help='Only update the receptor')
    parser.add_argument('-o', '--outprefix', type=str, help="Prefix for output files,"
    " default dream_<model>.<pid>", default='')
    parser.add_argument('-g', '--gpu',type=int, help='Specify GPU to run on', default=-1)
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate"
            " for input grid updates, default=0.1.")
    parser.add_argument('--threshold', type=float, default=1e-5,
            help="Convergence threshold for early termination, default 1e-5")
    parser.add_argument('-an', '--allow_negative', default=False,
            action='store_true', help="Allow negative density, useful if the"
            "result is to be used for a similarity search rather than to"
            "represent a physical molecule")
    parser.add_argument('-a', '--dump_all', default=False, action='store_true',
            help='Dump all intermediate grids from optimization, not just the \
            first and last')

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

#retrieve recmap, ligmap, also batch size if needed
#added a force_backward option to train.py which should suffice to generate the
#input diff blob, don't need to worry about setting lr_mult or decay_mult since
#we aren't using a solver
#make sure no random translation/rotation is added
resolution = 0.0
dimension = 0.0
recmap = ''
ligmap = ''
netparam = NetParameter()
with open(args.model) as f:
    prototxt.Merge(f.read(), netparam)
for layer in netparam.layer:
    if layer.type == "MolGridData":
        ligmap = layer.molgrid_data_param.ligmap
        recmap = layer.molgrid_data_param.recmap
        resolution = layer.molgrid_data_param.resolution
        dimension = layer.molgrid_data_param.dimension
        if not args.batch_size:
            args.batch_size = layer.molgrid_data_param.batch_size
        layer.molgrid_data_param.random_translate = 0
        layer.molgrid_data_param.random_rotation = False
tmpmodel = 'tmp.prototxt'
with open(tmpmodel, 'w') as f:
    f.write(str(netparam))

#we'll process these in batch-sized chunks
#at the beginning of each batch we'll do a forward pass through the full net
#after that we'll go forward starting from the second layer until we're done
#with that batch and start over with the next one
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
    write_model_file(test_model, tmpmodel, train_file, train_file, args.data_root, True)

    #construct net and update weights
    check_file_exists(args.weights)
    check_file_exists(test_model)
    net = caffe.Net(test_model, caffe.TRAIN, weights=args.weights)
    found = 0
    for i,layer in enumerate(net.layers):
        if layer.type == 'MolGridData':
            mgrid = layer
            mgrid_name = net._layer_names[i]
            found = 1
        elif found:
            next_name = net._layer_names[i]
            break
 
    #figure out which channels to exclude from updates if applicable
    #rec channels always come first
    channel_list = []
    if recmap:
        channel_list += get_channel_list(recmap, "Rec_")
        nrec_channels = count_lines(recmap)
    else:
        channel_list += caffe.get_rec_types(mgrid)
        nrec_channels = len(channel_list)
    if ligmap:
        channel_list += get_channel_list(ligmap, "Lig_")
        nlig_channels = count_lines(ligmap)
    else:
        channel_list += caffe.get_lig_types(mgrid)
        nlig_channels = len(channel_list) - nrec_channels

    nexamples = count_lines(train_file)
    nchunks = int(math.ceil((float(nexamples)) / args.batch_size))
    struct_names = get_structure_names(train_file)
    for chunk in range(nchunks):
        startline = chunk * args.batch_size
        diffs = []
        all_y_scores = []
        #do forward, backward, update input grid for desired number of iters 
        #TODO: momentum? otherwise change the update (switch to BFGS?)
        for i in xrange(args.iterations):
            if i == 0:
                startlayer = mgrid_name
            else:
                startlayer = next_name
            res = net.forward(start=startlayer)
            assert 'output' in res, "Network must produce output"
            all_y_scores.append([float(x[1]) for x in res['output']])
            #if there's a max pool before the first conv/inner product layer,
            #switch it to ave pool for backward, then switch it back
            switched = caffe.toggle_max_to_ave(net)
            net.backward()
            if switched:
                caffe.toggle_ave_to_max(net)
            current_grid = net.blobs['data'].data
            grid_diff = net.blobs['data'].diff
            #TODO: assumes second dim is atom type channel
            norm = np.linalg.norm(grid_diff)
            diffs.append(norm)
            if norm < args.threshold:
                print "gradient sufficiently converged, terminating early\n"
                break
            assert len(current_grid.shape) == 5, "Currently require grids to be NxCxDIMxDIMxDIM"
            assert len(grid_diff.shape) == 5, "Currently require grids to be NxCxDIMxDIMxDIM"
            dim = current_grid.shape[-1]
            if args.exclude_receptor:
                current_grid[:,nrec_channels:,:,:,:] -= args.lr * grid_diff[:,nrec_channels:,:,:,:]
            elif args.exclude_ligand:
                current_grid[:,:nrec_channels,:,:,:] -= args.lr * grid_diff[:,nrec_channels:,:,:,:]
            else:
                current_grid -= args.lr * grid_diff
            #don't let anything go negative, if desired
            if not args.allow_negative:
                current_grid[current_grid < 0] = 0
            #dump a grid at every iteration if desired, otherwise just dump the
            #first and last time
            if (i == 0) or (i == (args.iterations-1)) or (args.dump_all):
                for ex in range(args.batch_size):
                    struct = struct_names[startline + ex]
                    resultname = '%s_iter%d' %(struct, i)
                    center = []
                    for layer in net.layers:
                        if layer.type == "MolGridData":
                            center = caffe.get_grid_center(mgrid, ex)
                    if not center:
                        raise ValueError("Unable to determine grid center")
                    dump_grid_dx(resultname,
                            net.blobs['data'].data[ex,...], dim, resolution,
                            dimension, center, channel_list)
        #do final evaluation, write to output files
        res = net.forward(start=next_name)
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
os.remove(tmpmodel)
