#!/usr/bin/env python

'''Takes a bunch of types training files. First argument is what index the receptor starts on
(ligands are assumed to be right after).  Reads in the gninatypes files specified in these types 
files and writes out two monolithic receptor and ligand cache files in version 2 format.

Version 2 is optimized for memory mapped storage of caches.  keys (file names) are stored
first followed by dense storage of values (coordinates and types).
'''

import os, sys
import struct, argparse


def writemol(fname, out):
    '''fname is full path of is gninatypes file, write it in the appropriate binary format to out'''
    try:
        with open(fname,'rb') as gninatype:
            data = gninatype.read()
            assert(len(data) % 16 == 0)
            natoms = len(data)/16
            out.write(struct.pack('i',natoms))
            out.write(data)            
    except Exception as e:
        print(fname)
        print(e)
                
def create_cache2(molfiles, data_root, outfile):
    '''Create an outfile molcache2 file from the list molfiles stored at data_root.'''
    out = open(outfile,'wb')
    #first byte is for versioning
    out.write(struct.pack('i',-1))
    out.write(struct.pack('L',0)) #placeholder for offset to keys
    
    offsets = dict() #indxed by mol, location of data
    #start writing molecular data
    for mol in molfiles:
        fname = mol
        if len(data_root):
            fname = data_root+'/'+mol
        offsets[mol] = out.tell()
        writemol(fname, out)
        
    start = out.tell() #where the names start
    for mol in molfiles:
        if len(mol) > 255:
            print("Skipping",mol,"since filename is too long")
            continue
        s = bytes(mol)
        out.write(struct.pack('B',len(s)))
        out.write(s)
        out.write(struct.pack('L',offsets[mol]))
        
    #now set start
    out.seek(4)
    out.write(struct.pack('L',start))
    out.seek(0,os.SEEK_END)
    out.close()
    
    

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--col', required=True,type=int,help='Column receptor starts on')
parser.add_argument('--recmolcache', default='rec.molcache2',type=str,help='Filename of receptor cache')
parser.add_argument('--ligmolcache', default='lig.molcache2',type=str,help='Filename of ligand cache')
parser.add_argument('-d','--data_root',type=str,required=False,help="Root folder for relative paths in train/test files",default='')
parser.add_argument('fnames',nargs='+',type=str,help='types files to process')

args = parser.parse_args()

#load all file names into memory
seenlig = set()
seenrec = set()
for fname in args.fnames:
    for line in open(fname):
        vals = line.split()
        rec = vals[args.col]
        ligs = vals[args.col+1:]
        
        if rec not in seenrec:
            seenrec.add(rec)
            
        for lig in ligs:
            if lig == '#' or lig.startswith('#'):
                break
            if lig not in seenlig:
                seenlig.add(lig)

create_cache2(list(seenrec), args.data_root, args.recmolcache)
create_cache2(list(seenlig), args.data_root, args.ligmolcache)


