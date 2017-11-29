#!/usr/bin/env python

'''quick script for generating a real model and caffe time command'''
import argparse,sys

parser = argparse.ArgumentParser(description='Train neural net on .types data.')
parser.add_argument('-m','--model',type=str,required=True,help="Model template. Must use TRAINFILE and TESTFILE")
parser.add_argument('-p','--prefix',type=str,required=True,help="Prefix for training/test files: <prefix>[train|test][num].types")
parser.add_argument('-o','--output',type=str,help="Output model (default timeit.model)",default="timeit.model")

args = parser.parse_args()

model = open(args.model).read()
model = model.replace('TRAINFILE','%strain0.types'%args.prefix)
model = model.replace('TESTFILE','%stest0.types'%args.prefix)

out = open(args.output,'w')
out.write(model)
print "caffe time -gpu 0 -model %s"%args.output

