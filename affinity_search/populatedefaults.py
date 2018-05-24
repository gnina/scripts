#!/usr/bin/env python

'''Given a results.csv file and password for db, put the contents into google sql'''

import sys, re, MySQLdb
import pandas as pd
import makemodel
import numpy as np

        
conn = MySQLdb.connect (host = "35.196.158.205",user = "opter",passwd=sys.argv[1],db="opt2")
cursor = conn.cursor()

opts = makemodel.getoptions()
params = makemodel.getdefaults()


params['id'] = 'REQUESTED'

#do 5 variations
for split in xrange(5):
    params['split'] = split
    params['seed'] = np.random.randint(0,100000)
    data = pd.DataFrame([params])
    row = data.iloc[0]
    insert = 'INSERT INTO params (%s) VALUES (%s)' % (','.join(row.index),','.join(['%s']*len(row)))
    cursor.execute(insert,row)
    
conn.commit()
