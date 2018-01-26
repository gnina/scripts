#!/usr/bin/env python
'''Connects to sql database.
   Checks (atomically to see if there are an configurations that should be run 
   because they are requested (R).  If so, runs one
   '''

#https://hyperopt-186617.appspot.com

import sys, re, MySQLdb, argparse, socket, tempfile
import pandas as pd
import numpy as np
import makemodel
import subprocess, os, json
from MySQLdb.cursors import DictCursor

parser = argparse.ArgumentParser(description='Run a configuration as part of a search')
parser.add_argument('--data_root',type=str,help='Location of gninatypes directory',default='')
parser.add_argument('--prefix',type=str,help='gninatypes prefix, needs to be absolute',required=True)
parser.add_argument('--host',type=str,help='Database host',required=True)
parser.add_argument('-p','--password',type=str,help='Database password',required=True)
parser.add_argument('--db',type=str,help='Database name',required=True)

args = parser.parse_args()

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))
    
def rm(inprogressname):
    try:
        os.remove(inprogressname)
    except OSError:
        pass
        
sys.path.append(get_script_path())
sys.path.append(get_script_path()+'/..') #train

def getcursor():
    '''create a connection and return a cursor;
    doing this guards against dropped connections'''
    conn = MySQLdb.connect (host = args.host,user = "opter",passwd=args.password,db=args.db)
    conn.autocommit(True)
    cursor = conn.cursor(DictCursor)
    return cursor

def getgpuid():
    '''return unique id of gpu 0'''
    gpuid = '0000'
    try:
        output = subprocess.check_output('nvidia-smi',shell=True,stderr=subprocess.STDOUT)
        m = re.search(r'00000:(\S\S:\S\S.\S) ',output)
        if m:
            gpuid = m.group(1)
    except Exception as e:
        print e
        print "Error accessing gpu"
        sys.exit(1)
    return gpuid

opts = makemodel.getoptions()
cursor = getcursor()

host = socket.gethostname() 

# determine a configuration to run
configs = None  #map from name to value

# check for an in progress file
inprogressname = '%s-%s-INPROGRESS' % (host,getgpuid())
print inprogressname

if os.path.isfile(inprogressname):
    config = json.load(open(inprogressname))
    d = config['msg']    
    #only retry once - remove the file
    print "Retrying with config: %s" % json.dumps(config)
    rm(inprogressname)
else:
#are there any requested configurations?  if so select one
    cursor.execute('SELECT * FROM params WHERE id = "REQUESTED"')
    rows = cursor.fetchall()
    config = None
    for row in rows:
        # need to atomically update id
        ret = cursor.execute('UPDATE params SET id = "INPROGRESS" WHERE serial = %s',[row['serial']])
        if ret: # success!
            #set config
            config = row
            break
            
    if config: #write out what we're doing
        d = tempfile.mkdtemp(prefix=socket.gethostname() +'-',dir='.')
        config['msg'] = d
        progout = open(inprogressname,'w')
        if 'time' in config:
            del config['time']
        progout.write(json.dumps(config))
        progout.close()


if not config:
    print "Nothing requested"
    sys.exit(2)  # there was nothing to do, perhaps we should shutdown?
    
#at this point have a configuration
values = ['0','0']
for (name,val) in sorted(opts.items()):
    values.append(str(config[name]))
    
cmdline = '%s/runline.py --prefix %s --data_root "%s" --seed %d --split %d --dir %s --line "%s"' % \
        (get_script_path(), args.prefix,args.data_root,config['seed'],config['split'], config['msg'], ' '.join(values))
print cmdline

#call runline to insulate ourselves from catestrophic failure (caffe)
try:
    output = subprocess.check_output(cmdline,shell=True,stderr=subprocess.STDOUT)
    d, R, rmse, auc, top = output.rstrip().split('\n')[-1].split()
except Exception as e:
    pid = os.getpid()
    out = open('output.%s.%d'%(host,pid),'w')
    if isinstance(e, subprocess.CalledProcessError):
        output = e.output
    out.write(output)
    cursor = getcursor()
    cursor.execute('UPDATE params SET id = "ERROR", msg = %s WHERE serial = %s',(str(pid),config['serial']))
    print "Error"
    print output
    if re.search(r'out of memory',output) and host.startswith('gnina'):
        #host migration restarts don't seem to bring the gpu up in agood state
        os.system("sudo reboot")
    rm(inprogressname)    
    sys.exit(0)  #we tried
    

#if successful, store in database

config['rmse'] = float(rmse)
config['R'] = float(R)
config['top'] = float(top)
config['auc'] = float(auc)
config['id'] = d
config['msg'] = 'SUCCESS'

serial = config['serial']
del config['serial']
sql = 'UPDATE params SET {} WHERE serial = {}'.format(', '.join('{}=%s'.format(k) for k in config),serial)
cursor = getcursor()
cursor.execute(sql, config.values())

rm(inprogressname)

