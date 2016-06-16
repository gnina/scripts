#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, re, sklearn, collections, argparse, sys, os
import sklearn.metrics
import caffe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train neural net on binmap data.')
    parser.add_argument('-m','--model',type=str,required=True,help="Model template. Must use TESTFILE with unshuffled, unbalanced input")
    parser.add_argument('-w','--weights',type=str,required=True,help="Model weights (.caffemodel)")
    parser.add_argument('-i','--input',type=str,required=True,help="Input binmaps to predict")
    parser.add_argument('-g','--gpu',type=int,help='Specify GPU to run on',default=-1)
    parser.add_argument('-o','--output',type=str,help='Output file name',default='-')
    parser.add_argument('-k','--keep',action='store_true',default=False,help="Don't delete prototxt files")
    
    args = parser.parse_args()
    
    if args.gpu >= 0:
        caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    
    model = open(args.model).read().replace('TESTFILE',args.input)
    #very obnoxiously, python interface requires network definition to be in a file
    testfile = 'predict.%d.prototxt' % os.getpid()
    with open(testfile,'w') as f:
        f.write(model)    

    net = caffe.Net(testfile, args.weights, caffe.TEST)
        
    if args.output == '-':
        out = sys.stdout
    else:
        out = open(args.output,'w')
    
    ytrue = []
    yscore = []
    for line in open(args.input):
        predict = net.forward()['output']
        if predict.shape[0] != 1:
            print "Error: require single-sized batches"
            sys.exit(1)
        
        out.write('%f %s' % (predict[0][1],line))
        ytrue.append(float(line.split()[0]))
        yscore.append(predict[0][1])
    
    if not args.keep:
        os.remove(testfile)

    auc = sklearn.metrics.roc_auc_score(ytrue,yscore)
    out.write("# AUC %f\n" % auc)
    
    
