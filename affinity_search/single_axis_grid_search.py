#!/usr/bin/env python

'''Given desired set of parameters, generates all configurations 
obtained by enumerate each parameter individually (continuous are discretized).

'''

import sys, re, MySQLdb, argparse, os, json, subprocess
import pandas as pd
import makemodel
import numpy as np
from MySQLdb.cursors import DictCursor
from outputjson import makejson
from populaterequests import addrows

    
parser = argparse.ArgumentParser(description='Exhaustive grid search along single axes of variation')
parser.add_argument('--host',type=str,help='Database host')
parser.add_argument('-p','--password',type=str,help='Database password')
parser.add_argument('--db',type=str,help='Database name',default='database')
parser.add_argument('-o','--output',type=str,help="Output file",default="rows.txt")
parser.add_argument('--parameters',type=file,help='parameters to enumerate',required=True)
args = parser.parse_args()

#get options
defaults = makemodel.getdefaults()
options = makemodel.getoptions()
opts = sorted(options.items())

#read in  list of parameters
params = args.parameters.read().rstrip().split()

outrows = set() #uniq configurations only (e.g., avoid replicating the default over and over again)
for param in params:
    if param in options:
        choices = options[param]
        if isinstance(choices, makemodel.Range):
            choices = np.linspace(choices.min,choices.max, 9)
        #for each parameter value, create a row
        for val in choices:
            row = ['P','P'] #spearmint
            for (name,_) in opts:
                if name == param:
                    row.append(val)
                else:
                    row.append(defaults[name])
            outrows.add(tuple(row))
                    
out = open(args.output,'w')
for row in outrows:
    out.write(' '.join(map(str,row))+'\n')

out.close()

if args.host:
    addrows(args.output,args.host,args.db,args.password)

    
    
