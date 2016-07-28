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
    model = open(args.model).read().replace('TESTFILE',args.input)
    #very obnoxiously, python interface requires network definition to be in a file
    testfile = 'predict.%d.prototxt' % os.getpid()
    with open(testfile,'w') as f:
        f.write(model)    
    net = caffe.Net(testfile, args.weights, caffe.TEST)
    output =[]
    for line in open(args.input):
        predict = net.forward()['output']
        if predict.shape[0] != 1:
            print "Error: require single-sized batches"
            sys.exit(1)
        
        output.append('%f %s' % (predict[0][1],line))
    if args.max_score: output = maxLigandScore(output)
    if not args.keep:
        os.remove(testfile)
    return output

def maxLigandScore(output):
	#output format: score label paths 
	ligands ={}
	for line in output:
		data = line.split(" ",2)
		score = data[0]
		true_class = data[1]
		path = data [2] # assumes that it contains [folder]/[ligname]_[#].gninatypes
		targetname= path.split("/",1)[0]
		ligname = path.rsplit("/",1)[-1]
		ligname = ligname.split(".gninatypes")[0]
		ligname = ligname.split("_")[0]
		key = targetname+ligname
		if key in ligands:
			if ligands[key][0] < score: ligands[key]=(score, '%s %s'%(true_class,path))
		else: ligands[key]=(score, '%s %s'%(true_class,path))
	new_output =ligands.values()
	for i in xrange(len(new_output)):
		new_output[i]= '%s %s'%(new_output[i][0],new_output[i][1])
	return new_output
	

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train neural net on binmap data.')
    parser.add_argument('-m','--model',type=str,required=True,help="Model template. Must use TESTFILE with unshuffled, unbalanced input. EX: file.model ")
    parser.add_argument('-w','--weights',type=str,required=True,help="Model weights (.caffemodel)")
    parser.add_argument('-i','--input',type=str,required=True,help="Input .types file to predict")
    parser.add_argument('-g','--gpu',type=int,help='Specify GPU to run on',default=-1)
    parser.add_argument('-o','--output',type=str,help='Output file name',default='-')
    parser.add_argument('-k','--keep',action='store_true',default=False,help="Don't delete prototxt files")
    parser.add_argument('--max_score',action='store_true',default=False,help="take max score per ligand as its score")
    parser.add_argument('--notcalc_predictions', type=str, default='',help='file of predictions')

    args = parser.parse_args()
    if args.output == '-':
        out = sys.stdout
    else:
        out = open(args.output,'w')
    if args.notcalc_predictions == '': output = predict(args)
    else:
		predictions=[]
		for line in open(args.notcalc_predictions).readlines():
			predictions.append(line)
		if args.max_score: output=maxLigandScore(predictions)
    out.writelines('%s'%line for line in output)
    
    
    

    
    
    
