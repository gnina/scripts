#!/usr/bin/env python

'''Return the top and R statistics for every row of the database that has them'''

import sys, re, MySQLdb, argparse, os, json, subprocess
import pandas as pd
import makemodel
import numpy as np

def getcursor():
    '''create a connection and return a cursor;
    doing this guards against dropped connections'''
    conn = MySQLdb.connect (host = args.host,user = "opter",passwd=args.password,db=args.db)
    conn.autocommit(True)
    cursor = conn.cursor()
    return cursor
    
    
parser = argparse.ArgumentParser(description='Return top and R statistics for successful rows in database')
parser.add_argument('--host',type=str,help='Database host',required=True)
parser.add_argument('-p','--password',type=str,help='Database password',required=True)
parser.add_argument('--db',type=str,help='Database name',default='opt1')

args = parser.parse_args()

cursor = getcursor()
cursor.execute('SELECT serial,top,R FROM params WHERE top IS NOT NULL')
rows = cursor.fetchall()
for row in rows:
    print '%d %f %f' % row
