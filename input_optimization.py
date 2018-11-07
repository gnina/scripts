#!/usr/bin/env python

import caffe
from train import write_model_file, check_file_exists
import argparse,glob,re,os,sys

'''
Use trained network to optimize grids associated with input structures. Need a
model template, data, and pre-trained weights. Input model template needs
lr_mult==0 for non-data layers, dream==True for molgrid layer, and no TEST phase.
'''

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
    parser.add_argument('-o', '--outprefix', type=str, help="Prefix for output files,"
    " default dream_<model>.<pid>", default='')
    parser.add_argument('-g', '--gpu',type=int, help='Specify GPU to run on', default=-1)
    parser.add_argument('--dynamic', action='store_true', default=False, help='Attempt'
    ' to adjust the base_lr in response to training progress')
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate"
            " for input grid updates.")

args = parser.parse_args()
pid = os.getpid()

if args.gpu >= 0:
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()

glob_files = glob.glob(args.prefix + '*')
train_files = []
test_file = ''
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

#TODO: check/enforce that lr_mult==0 for non-data layers in the dream net template?
#check/enforce that molgrid layer has dream==True?
for train_file in train_files:
    test_model = 'opt.%d.prototxt' % pid
    write_model_file(test_model, args.model, train_files, test_file, args.data_root, True)

    #construct net and update weights
    check_file_exists(args.weights)
    check_file_exists(args.trained_net)
    trained_net = caffe.Net(args.trained_net, args.weights, caffe.TRAIN)
    dream_net = caffe.Net(test_model, caffe.TRAIN)
    for layer in trained_net._layer_names:
        if layer in dream_net._layer_names:
            dream_net.params[layer][0].data[...] = trained_net.params[layer][0].data[...]
            dream_net.params[layer][1].data[...] = trained_net.params[layer][1].data[...]
    
    #output final prediction/AUC to see the final result of input optimization
    train = {'aucs':[], 'y_true':[], 'y_score':[], 'losses':[], 'rmsds':[],
            'y_aff':[], 'y_predaff':[], 'rmsd_rmses':[]}
                    
    #do forward, backward, update input grid for desired number of iters 
    #(molgrid currently handles dumping)
    #TODO: momentum?
    for i in xrange(args.iterations):
        dream_net.forward()
        dream_net.backward()
        current_grid = dream_net.blobs['data'].data
        grid_diff = dream_net.blobs['data'].diff
        current_grid -= args.lr * grid_diff
        dream_net.blobs['data'].data[...] = current_grid
