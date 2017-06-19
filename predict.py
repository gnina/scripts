#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, re, sklearn, collections, argparse, sys, os
import sklearn.metrics
import caffe
from caffe.proto.caffe_pb2 import NetParameter
import google.protobuf.text_format as prototxt
from train import evaluate_test_net


def write_model_file(model_file, template_file, test_file, root_folder):
    param = NetParameter()
    with open(template_file, 'r') as f:
        prototxt.Merge(f.read(), param)
    for layer in param.layer:
        if layer.molgrid_data_param.source == 'TESTFILE':
            layer.molgrid_data_param.source = test_file
        if layer.molgrid_data_param.root_folder == 'DATA_ROOT':
            layer.molgrid_data_param.root_folder = root_folder
    with open(model_file, 'w') as f:
        f.write(str(param))


def predict(args):
    if args.gpu >= 0:
        caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    test_model = 'predict.%d.prototxt' % os.getpid()
    write_model_file(test_model, args.model, args.input, args.data_root)
    test_net = caffe.Net(test_model, args.weights, caffe.TEST)
    with open(args.input, 'r') as f:
        lines = f.readlines()
    result, _ = evaluate_test_net(test_net, len(lines), 1, 0)
    auc, y_true, y_score, loss, rmsd, y_affinity, y_predaff = result
    assert np.all(y_true == [float(l.split(' ')[0]) for l in lines]) #check alignment
    if args.affinity:
        assert len(y_predaff) > 0
        predict = y_predaff
    else:
        predict = y_score
    output_lines = ['%f %s' % t for t in zip(predict, lines)]
    if args.max_score:
        output_lines = maxLigandScore(output_lines)
    if not args.keep:
        os.remove(test_model)
    return output_lines


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
    #output format: score label [affinity] rec_path pose_path
    ligands = {}
    for line in lines:
        data = line.split(' ')
        score = float(data[0])
        label = float(data[1])
        try:
            affinity = float(data[2])
            rec_path = data[3].strip()
            pose_path = data[4].strip()
        except ValueError:
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
    parser.add_argument('-a','--affinity',default=False,action='store_true',required=False,help='Predict affinity instead of active/decoy label')
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
    if args.affinity:
        y_predaff = []
        y_true = []
        y_affinity = []
        for line in predictions:
            data = line.split(' ')
            y_predaff.append(float(data[0]))
            y_true.append(float(data[1]))
            y_affinity.append(float(data[2]))
        if len(np.unique(y_true)) > 1:
            y_predaff = np.array(y_predaff)
            y_affinity = np.array(y_affinity)
            yt = np.array(y_true, np.bool)
            rmsd = sklearn.metrics.mean_squared_error(y_affinity[yt], y_predaff[yt])
            out.write("# RMSD %.5f\n" % rmsd)
    else:
        y_score = []
        y_true = []
        for line in predictions:
            data = line.split(' ')
            y_score.append(float(data[0]))
            y_true.append(float(data[1]))
        if len(np.unique(y_true)) > 1:
            auc = sklearn.metrics.roc_auc_score(y_true, y_score)
            out.write("# AUC %.5f\n" % auc)

