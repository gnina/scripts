#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, re, sklearn, collections, argparse, sys, os
import sklearn.metrics
import caffe


def predict(args):
    if args.gpu >= 0:
        caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    model = open(args.model).read().replace('TESTFILE', args.input).replace('DATA_ROOT', args.data_root)
    #very obnoxiously, python interface requires network definition to be in a file
    testfile = 'predict.%d.prototxt' % os.getpid()
    with open(testfile, 'w') as f:
        f.write(model)
    net = caffe.Net(testfile, args.weights, caffe.TEST)
    output = []
    for line in open(args.input):
	out = net.forward()
	if 'output' in out:
	    predict = out['output'][0][1]
	elif 'predaff' in out:
	    predict = out['predaff']
        elif 'rankoutput' in out:
            predict = out['rankoutput']
        output.append('%f %s' % (predict, line))
    if args.max_score:
        output = maxLigandScore(output)
    if not args.keep:
        os.remove(testfile)
    return output


def get_ligand_key(rec_path, pose_path):
    # no good naming convention, so just use the receptor name
    # and each numeric part of the ligand/pose name except for
    # the last, which is the pose number of the ligand
    rec_dir = os.path.dirname(rec_path)
    rec_name = rec_dir.rsplit('/', 1)[-1]
    pose_name = os.path.splitext(os.path.basename(pose_path))[0]
    pose_name_nums = []
    for i, part in enumerate(pose_name.split('_')):
        try:
            pose_name_nums.append(int(part))
        except ValueError:
            continue
    return tuple([rec_name] + pose_name_nums[:-1])


def maxLigandScore(lines):
    #output format: score label paths 
    ligands = {}
    for line in lines:
        data = line.split(' ')
        score = float(data[0])
        rec_path = data[2].strip()
        pose_path = data[3].strip()
        key = get_ligand_key(rec_path, pose_path)
        if key not in ligands or score > ligands[key][0]:
            ligands[key] = (score, line)
    return [ligands[key][1] for key in ligands]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Test neural net on gninatypes data.')
    parser.add_argument('-m','--model',type=str,required=True,help="Model template. Must use TESTFILE with unshuffled, unbalanced input. EX: file.model ")
    parser.add_argument('-w','--weights',type=str,required=True,help="Model weights (.caffemodel)")
    parser.add_argument('-d','--data_root',type=str,required=False,help="Root folder for paths in .types files",default='')
    parser.add_argument('-i','--input',type=str,required=True,help="Input .types file to predict")
    parser.add_argument('-g','--gpu',type=int,help='Specify GPU to run on',default=-1)
    parser.add_argument('-o','--output',type=str,help='Output file name',default=None)
    parser.add_argument('-k','--keep',action='store_true',default=False,help="Don't delete prototxt files")
    parser.add_argument('--max_score',action='store_true',default=False,help="take max score per ligand as its score")
    parser.add_argument('--notcalc_predictions', type=str, default='',help='use file of predictions instead of calculating')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    if not args.output:
        out = sys.stdout
    else:
        out = open(args.output, 'w')
    if not args.notcalc_predictions:
        predictions = predict(args)
    else:
        with open(args.notcalc_predictions, 'r') as f:
            predictions = f.readlines()
        if args.max_score:
            predictions = maxLigandScore(predictions)
    out.writelines(predictions)
    #add auc to end of file
    ytrue = []
    yscore = []
    for line in predictions:
        data = line.split(' ')
        ytrue.append(float(data[1]))
        yscore.append(float(data[0]))
    if len(np.unique(ytrue)) > 1:
        auc = sklearn.metrics.roc_auc_score(ytrue, yscore)
        out.write("# AUC %.2f\n" % auc)

