#!/usr/bin/env python3

import predict
import sklearn.metrics
import argparse, sys
import os
import numpy as np
import glob
import re
import matplotlib.pyplot as plt

def calc_auc(predictions):
	y_true =[]
	y_score=[]
	for line in predictions:
		values= line.split(" ")
		y_true.append(float(values[1]))
		y_score.append(float(values[0]))
	auc = sklearn.metrics.roc_auc_score(y_true,y_score)
	return auc

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='bootstrap(sampling with replacement) test')
	parser.add_argument('-m','--model',type=str,required=True,help="Model template. Must use TESTFILE with unshuffled, unbalanced input")
	parser.add_argument('-w','--weights',type=str,required=True,help="Model weights (.caffemodel)")
	parser.add_argument('-i','--input',type=str,required=True,help="Input .types file to predict")
	parser.add_argument('-g','--gpu',type=int,help='Specify GPU to run on',default=-1)
	parser.add_argument('-o','--output',type=str,default='',help='Output file name,default= predict_[model]_[input]')
	parser.add_argument('--iterations',type=int,default=1000,help="number of times to bootstrap")
	parser.add_argument('-k','--keep',action='store_true',default=False,help="Don't delete prototxt files")
	parser.add_argument('-n', '--number',action='store_true',default=False,help="if true uses caffemodel/input as is. if false uses all folds")
	parser.add_argument('--max_score',action='store_true',default=False,help="take max score per ligand as its score")
	parser.add_argument('--notcalc_predictions', type=str, default='',help='file of predictions')
	args = parser.parse_args()
	if args.output == '':
		output = 'bootstrap_%s_%s'%(args.model, args.input)
	else:
		output = args.output
	outname=output	
	predictions=[]
	if args.notcalc_predictions=='':
		cm = args.weights
		ts = args.input
		if not args.number:
			foldnum = re.search('.[0-9]_iter',cm).group()
			cm=cm.replace(foldnum, '.[0-9]_iter')
			foldnum = re.search('[0-9].types',ts).group()
			ts=ts.replace(foldnum, '[NUMBER].types')
		
		for caffemodel in glob.glob(cm):
			testset = ts
			if not args.number:
				num = re.search('.[0-9]_iter',caffemodel).group()
				num=re.search(r'\d+', num).group()
				testset = ts.replace('[NUMBER]',num)
			args.input = testset
			args.weights = caffemodel
			predictions.extend(predict.predict_lines(args))
	elif args.notcalc_predictions != '':
		for line in open(args.notcalc_predictions).readlines():
			predictions.append(line)
		
	all_aucs=[]
	for _ in range(args.iterations):
		sample = np.random.choice(predictions,len(predictions), replace=True)
		all_aucs.append(calc_auc(sample))
	mean=np.mean(all_aucs)
	std_dev = np.std(all_aucs)
	txt = 'mean: %.2f standard deviation: %.2f'%(mean,std_dev)
	print(txt)
	output = open(output, 'w')
	output.writelines('%.2f\n' %auc for auc in all_aucs)
	output.write(txt)
	output.close()

	plt.figure()
	plt.boxplot(all_aucs,0,'rs',0)
	plt.title('%s AUCs'%args.output, fontsize=22)
	plt.xlabel('AUC(%s)'%txt, fontsize=18)
	plt.savefig('%s_plot.pdf'%outname,bbox_inches='tight')



	
