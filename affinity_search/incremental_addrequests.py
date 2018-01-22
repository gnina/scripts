#!/usr/bin/env python

'''Checks an sql database to determine what jobs to run next for
hyperparameter optimization.

Works incrementally by taking a prioritized order for evaluating
parameters.  It is assumed the database is already populated with at least
one good model.  The best model is identified according to some metric
(I'm thinking R, or maybe top*R - an average of identical models is taken).  The parameters
for this model become the defaults.  Then, in priority order, the i'th parameter is considered
 We compoute the average metric for all evaluated models.  We ignore models that don't
 have a required number of minimum evaluations
 We check if the metric has improved on the previous last best
 If it hasn't:
    All paramters > i are set so they only have defaults as options in the spearmint config
    Any result rows that do not match the defaults for parameters >i are omitted.
    We run spearmint to get new suggestions and add them to the database
 Otherwise 
   We increment i and save information (best value at previous level
 
 Note this stores ongoing information in a file INCREMENTAL.info
 If this file doesn't exist we start from the beginning.
 

'''

import sys, re, MySQLdb, argparse, os, json, subprocess
import pandas as pd
import makemodel
import numpy as np
from MySQLdb.cursors import DictCursor
from outputjson import makejson
from populaterequests import addrows

def getcursor():
    '''create a connection and return a cursor;
    doing this guards against dropped connections'''
    conn = MySQLdb.connect (host = args.host,user = "opter",passwd=args.password,db=args.db)
    conn.autocommit(True)
    cursor = conn.cursor(DictCursor)
    return cursor
    
    
parser = argparse.ArgumentParser(description='Generate more configurations if needed')
parser.add_argument('--host',type=str,help='Database host',required=True)
parser.add_argument('-p','--password',type=str,help='Database password',required=True)
parser.add_argument('--db',type=str,help='Database name',default='opt1')
parser.add_argument('--pending_threshold',type=int,default=12,help='Number of pending jobs that triggers an update')
parser.add_argument('-n','--num_configs',type=int,default=4,help='Number of configs to generate - will add 3X as many jobs') 
parser.add_argument('-s','--spearmint',type=str,help='Location of spearmint-lite.py',required=True)
parser.add_argument('--model_threshold',type=int,default=32,help='Number of unique models to evaluate at a level before giving up and going to the next level')
parser.add_argument('--priority',type=file,help='priority order of parameters',required=True)
parser.add_argument('--info',type=str,help='incremental information file',default='INCREMENTAL.info')
parser.add_argument('--mingroup',type=int,help='required number of evaluations of a model for it to count',default=3)
args = parser.parse_args()

opts = makemodel.getoptions()


# first see how many id=REQUESTED jobs there are
cursor = getcursor()
cursor.execute('SELECT COUNT(*) FROM params WHERE id = "REQUESTED"')
rows = cursor.fetchone()
pending = rows.values()[0]

print "Pending jobs:",pending

#if more than pending_threshold, quit
if pending > args.pending_threshold:
    sys.exit(0)

#create gnina-spearmint directory if it doesn't exist already
if not os.path.exists('gnina-spearmint-incremental'):
    os.makedirs('gnina-spearmint-incremental')
    
#read in prioritized list of parameters
params = args.priority.read().rstrip().split()

#get data and compute average metric of each model
cursor.execute('SELECT * FROM params')
rows = cursor.fetchall()
data = pd.DataFrame(list(rows))
nonan = data.dropna('index')
grouped = nonan.groupby(params)
metrics = grouped.mean()[['R','top']]
metrics = metrics[grouped.size() >= args.mingroup]
metrics['Rtop'] = metrics.R * metrics.top
defaultparams = metrics['Rtop'].idxmax()  #this is in priority order
bestRtop = metrics['Rtop'].max()

#figure out what param we are on
if os.path.exists(args.info):
    #info file has what iteration we are on and the previous best when we moved to that iteration
    (level, prevbest) = open(args.info).readlines()[-1].split()
    level = int(level)
else:
    #very first time we've run
    level = 0
    prevbest = bestRtop
    info = open(args.info,'w')
    info.write('0 %f\n'%bestRtop)
    info.close()

#check to see if we should promote level
if bestRtop > prevbest:
    level += 1
    info = open(args.info,'a')
    info.write('%d %f\n',(level,bestRtop))
    info.close()
    
#create config.json without defaulted parameters
config = makejson()
defaults = dict()
for (i,(name,value)) in enumerate(zip(params,defaultparams)):
    if i > level:
        defaults[name] = value
        del config[name]
           
cout = open('gnina-spearmint-incremental/config.json','w')
cout.write(json.dumps(config, indent=4)+'\n')
cout.close()

#output results.data using top*R
#don't use averages, since in theory spearmint will use the distribution intelligently
#also include rows without values to avoid repetition

resf = open('gnina-spearmint-incremental/results.dat','w')
uniqconfigs = set()
validrows = 0
for (i,row) in data.iterrows():
    outrow = []
    for (name,vals) in sorted(opts.items()):
        if name == 'resolution':
            val = str(float(row[name])) #gets returned as 1 instead of 1.0 
        else:
            val = str(row[name])
            
        if name in defaults:  # is this row acceptable
            if type(row[name]) == float or type(row[name]) == int:
                if np.abs(float(defaults[name])-row[name]) > 0.00001:
                    break
            elif row[name] != defaults[name]:
                break
        else:
            outrow.append(val) 
    else:  #execute if we didn't break        
        validrows += 1
        uniqconfigs.add(tuple(config))
        Rtop = row['R']*row['top']
        if np.isfinite(Rtop):
            resf.write('%f 0 '%-Rtop)
        else:
            resf.write('P P ')
        #outrow is in opt order, but with defaults removed
        resf.write(' '.join(outrow))
        resf.write('\n')
resf.close()
        
gseed = len(uniqconfigs) #not clear this actually makes sense in our context..
# run spearmint-light, set the seed to the number of unique configurations
subprocess.call(['python',args.spearmint, '--method=GPEIOptChooser', '--grid-size=20000', 
        'gnina-spearmint-incremental', '--n=%d'%args.num_configs, '--grid-seed=%d' % gseed])

#get the generated lines from the file
lines = open('gnina-spearmint-incremental/results.dat').readlines()
newlines = np.unique(lines[validrows:])
print len(newlines),args.num_configs
assert(len(newlines) > 0)
out = open('gnina-spearmint-incremental/newrows.dat','w')
for line in newlines:
    vals = line.rstrip().split()
    pos = 2
    outrow = [vals[0],vals[1]]
    for (name,_) in sorted(opts.items()):
        if name in defaults:  
            outrow.append(defaults[name])
        else: #not defaults in opt order
            outrow.append(vals[pos])
            pos += 1
    assert(pos == len(vals))
    out.write(' '.join(outrow))
    out.write('\n')
out.close()
#add to database as REQUESTED jobs
#addrows('gnina-spearmint-incremental/newrows.dat',args.host,args.db,args.password)

