#!/usr/bin/env python

'''Return aggregated statistics for database'''

import sys, re, MySQLdb, argparse, os, json, subprocess
import pandas as pd
import makemodel
import numpy as np
from MySQLdb.cursors import DictCursor

def getcursor(host,passwd,db):
    '''create a connection and return a cursor;
    doing this guards against dropped connections'''
    conn = MySQLdb.connect (host = host,user = "opter",passwd=passwd,db=db)
    conn.autocommit(True)
    cursor = conn.cursor(DictCursor)
    return cursor
    
def __my_flatten_cols(self, how="_".join, reset_index=True):
    how = (lambda iter: list(iter)[-1]) if how == "last" else how
    self.columns = [how(filter(None, map(str, levels))) for levels in self.columns.values] \
                    if isinstance(self.columns, pd.MultiIndex) else self.columns
    return self.reset_index() if reset_index else self
pd.DataFrame.my_flatten_cols = __my_flatten_cols
    
def getres(host, password, db, mingroup, priority, selected_params):
    '''return dataframe grouped by all params with params selected out'''
    cursor = getcursor(host,password,db)
    cursor.execute('SELECT * FROM params')
    rows = cursor.fetchall()
    data = pd.DataFrame(list(rows))
    #make errors zero - appropriate if error is due to parameters
    data.loc[data.id == 'ERROR','R'] = 0
    data.loc[data.id == 'ERROR','rmse'] = 0
    data.loc[data.id == 'ERROR','top'] = 0
    data.loc[data.id == 'ERROR','auc'] = 0

    data['Rtop'] = data.R*data.top
    nonan = data.dropna('index').apply(pd.to_numeric, errors='ignore')

    #read in prioritized list of parameters
    params = open(priority).read().rstrip().split()

    grouped = nonan.groupby(params)
    metrics = grouped.agg([np.mean,np.std])
    metrics = metrics[grouped.size() >= mingroup]
    metrics = metrics.my_flatten_cols()

    metrics = metrics.reset_index()

    sel = ['rmse_mean','top_mean','R_mean','auc_mean','Rtop_mean','rmse_std','top_std','R_std','auc_std','Rtop_std']
    if selected_params:
        sel += selected_params
    return  metrics.loc[:,sel]

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Return aggregatedfor successful rows in database')
    parser.add_argument('--host',type=str,help='Database host',required=True)
    parser.add_argument('-p','--password',type=str,help='Database password',required=True)
    parser.add_argument('--db',type=str,help='Database name',default='opt1')
    parser.add_argument('--mingroup',type=int,help='required number of evaluations of a model for it to count',default=5)
    parser.add_argument('--priority',type=str,help='priority order of parameters',required=True,default="priority")
    parser.add_argument('-s','--selected_params',nargs='*',help='parameters whose values should be printed with metrics')

    args = parser.parse_args()
    
    metrics = getres(**vars(args))
    print metrics.to_csv(sep='\t',index_label='index')


