#!/usr/bin/env python

'''Given a results.dat file and password for db, put the contents into google sql as
configurations that are being requested.  Specify five of each.'''

import sys, re, MySQLdb
import pandas as pd
import makemodel
import numpy as np

opts = makemodel.getoptions()

def addrows(fname,host,db,password,start=0):
    '''add rows from fname into database, starting at row start'''
    data = pd.read_csv(fname,delim_whitespace=True,header=None)
    colnames = ['P1','P2']
    for (name,val) in sorted(opts.items()):
        colnames.append(name)
        
    data.columns = colnames
    data = data.drop(['P1','P2'],axis=1)

    conn = MySQLdb.connect (host = host,user = "opter",passwd=password,db=db)
    cursor = conn.cursor()


    for (i,row) in data[start:].iterrows():
        names = ','.join(row.index)
        values = ','.join(['%s']*len(row))
        names += ',id'
        values += ',"REQUESTED"'
        #do five variations
        for split in range(5):
            seed = np.random.randint(0,100000)
            n = names + ',split,seed'
            v = values + ',%d,%d' % (split,seed) 
            insert = 'INSERT INTO params (%s) VALUES (%s)' % (n,v)
            cursor.execute(insert,row)
    conn.commit()

if __name__ == '__main__':
    addrows(sys.argv[1],"35.196.158.205","opt2",sys.argv[2])
