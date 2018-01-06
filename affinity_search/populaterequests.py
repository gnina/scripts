#!/usr/bin/env python

'''Given a results.dat file and password for db, put the contents into google sql as
configurations that are being requested.  Specify three of each.'''

import sys, re, MySQLdb
import pandas as pd
import makemodel
import numpy as np

opts = makemodel.getoptions()


data = pd.read_csv(sys.argv[1],delim_whitespace=True,header=None)
colnames = ['P1','P2']
for (name,val) in sorted(opts.items()):
    colnames.append(name)
    
data.columns = colnames
data = data.drop(['P1','P2'],axis=1)

conn = MySQLdb.connect (host = "35.196.158.205",user = "opter",passwd=sys.argv[2],db="opt1")
cursor = conn.cursor()


for (i,row) in data.iterrows():
    names = ','.join(row.index)
    values = ','.join(['%s']*len(row))
    names += ',id'
    values += ',"REQUESTED"'
    #do three variations
    for _ in xrange(3):
        split = np.random.randint(0,5)
        seed = np.random.randint(0,100000)
        n = names + ',split,seed'
        v = values + ',%d,%d' % (split,seed) 
        insert = 'INSERT INTO params (%s) VALUES (%s)' % (n,v)
        cursor.execute(insert,row)
conn.commit()
