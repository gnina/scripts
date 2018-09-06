#!/usr/bin/env python

'''Takes a bunch of types training files. First argument is what index the receptor starts on
(ligand is assumed to be right after).  Reads in the gninatypes files specified in these types 
files and writes out two monolithic receptor and ligand cache files for use with recmolcache
and ligmolcache molgrid options'''

import os, sys
import struct, argparse

def writemol(root, mol, out):
    '''mol is gninatypes file, write it in the appropriate binary format to out'''
    fname = root+'/'+mol
    try:
        with open(fname,'rb') as gninatype:
            if len(fname) > 255:
                print "Skipping",mol,"since filename is too long"
                return
            s = bytes(mol)
            out.write(struct.pack('b',len(s)))
            out.write(s)
            data = gninatype.read()
            assert(len(data) % 16 == 0)
            natoms = len(data)/16
            out.write(struct.pack('i',natoms))
            out.write(data)            
    except Exception as e:
        print mol
        print e

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--col', required=True,type=int,help='Column receptor starts on')
parser.add_argument('--recmolcache', default='rec.molcache',type=str,help='Filename of receptor cache')
parser.add_argument('--ligmolcache', default='lig.molcache',type=str,help='Filename of ligand cache')
parser.add_argument('-d','--data_root',type=str,required=False,help="Root folder for relative paths in train/test files",default='')
parser.add_argument('fnames',nargs='+',type=str,help='types files to process')

args = parser.parse_args()

recout = open(args.recmolcache,'wb')
ligout = open(args.ligmolcache,'wb')

seenlig = set()
seenrec = set()
for fname in args.fnames:
    for line in open(fname):
        vals = line.split()
        rec = vals[args.col]
        lig = vals[args.col+1]
        
        if rec not in seenrec:
            seenrec.add(rec)
            writemol(args.data_root, rec, recout)
            
        if lig not in seenlig:
            seenlig.add(lig)
            writemol(args.data_root, lig, ligout)
    
