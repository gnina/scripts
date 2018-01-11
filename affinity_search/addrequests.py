#!/usr/bin/env python

'''Check the sql database to see if the number of pending jobs is below
a threshold.  If so, download the table and run spearmint twice, once for
top and once for R, each time generating N*3 new jobs.  For each of the
N configurations there are 3 variants with different splits and seeds.'''

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
parser.add_argument('-n','--num_configs',type=int,default=4,help='Number of configs to generate - will add 6X as many jobs') 
parser.add_argument('-s','--spearmint',type=str,help='Location of spearmint-lite.py',required=True)

args = parser.parse_args()

opts = makemodel.getoptions()


# first see how many id=REQUESTED jobs there are
cursor = getcursor()
cursor.execute('SELECT COUNT(*) FROM params WHERE id = "REQUESTED"')
rows = cursor.fetchone()
pending = rows.values()[0]
cursor.close()

print "Pending jobs:",pending

#if more than pending_threshold, quit
if pending > args.pending_threshold:
    sys.exit(0)

#create gnina-spearmint directory if it doesn't exist already
if not os.path.exists('gnina-spearmint'):
    os.makedirs('gnina-spearmint')
    
#create config.json
cout = open('gnina-spearmint/config.json','w')
cout.write(json.dumps(makejson(), indent=4)+'\n')
cout.close()

#for each of top and R
for metric in ['top','R']:
    #get the whole database
    cursor = getcursor()
    cursor.execute('SELECT * FROM params')
    rows = cursor.fetchall()
    resf = open('gnina-spearmint/results.dat','w')
    #write out a results.dat file, P for NULL metric, negated for real
    uniqconfigs = set()
    for row in rows:
        config = []
        for (name,vals) in sorted(opts.items()):
            if name == 'resolution':
                val = str(float(row[name])) #gets returned as 1 instead of 1.0 
            else:
                val = str(row[name])
            config.append(val)
        uniqconfigs.add(tuple(config))
        if row[metric]: #not null
            resf.write('%f 0 '%-row[metric]) #spearmint tries to _minimize_ so negate
        else:
            resf.write('P P ')
        resf.write(' '.join(config))
        resf.write('\n')
    resf.close()
    
    gseed = len(uniqconfigs)
    # run spearmint-light, set the seed to the number of unique configurations
    subprocess.call(['python',args.spearmint, '--method=GPEIOptChooser', '--grid-size=20000', 
            'gnina-spearmint', '--n=%d'%args.num_configs, '--grid-seed=%d' % gseed])
    print ['python',args.spearmint, '--method=GPEIOptChooser', '--grid-size=20000',
                        'gnina-spearmint', '--n=%d'%args.num_configs, '--grid-seed=%d' % gseed]
    #get the generated lines from the file
    lines = open('gnina-spearmint/results.dat').readlines()
    newlines = lines[len(rows):]
    print len(newlines),args.num_configs
    assert(len(newlines) > 0)
    print newlines
    #add to database as REQUESTED jobs
    addrows('gnina-spearmint/results.dat',args.host,args.db,args.password,start=len(rows))

