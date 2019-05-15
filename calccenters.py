#!/usr/bin/env python3

'''Glob through files in current directory looking for */*_ligand.sdf and */*.gninatypes (assuming PDBbind layout).
Calculate the distance between centers.  If types files are passed, create versions with this information,
optionally filtering. 
'''

import sys,glob,argparse,os
import numpy as np
import pybel
import struct
import openbabel

openbabel.obErrorLog.StopLogging()

parser = argparse.ArgumentParser()

parser.add_argument('typefiles',metavar='file',type=str, nargs='+',help='Types files to process')
parser.add_argument('--filter',type=float,default=100.0,help='Filter out examples greater the specified value')
parser.add_argument('--suffix',type=str,default='_wc',help='Suffix for new types files')
args = parser.parse_args()

centerinfo = dict()
#first process all gninatypes files in current directory tree
for ligfile in glob.glob('*/*_ligand.sdf'):
    mol = next(pybel.readfile('sdf',ligfile))
    #calc center
    center = np.mean([a.coords for a in mol.atoms],axis=0)
    dir = ligfile.split('/')[0]
    for gtypes in glob.glob('%s/*.gninatypes'%dir):
        buf = open(gtypes,'rb').read()
        n = len(buf)/4
        vals = np.array(struct.unpack('f'*n,buf)).reshape(n/4,4)
        lcenter = np.mean(vals,axis=0)[0:3]
        dist = np.linalg.norm(center-lcenter)
        centerinfo[gtypes] = dist

for tfile in args.typefiles:
    fname,ext = os.path.splitext(tfile)
    outname = fname+args.suffix+ext
    out = open(outname,'w')
    for line in open(tfile):
        lfile = line.split('#')[0].split()[-1]
        if lfile not in centerinfo:
            print("Missing",lfile,tfile)
            sys.exit(0)
        else:
            d = centerinfo[lfile]
            if d < args.filter:
                out.write(line.rstrip()+" %f\n"%d)        
