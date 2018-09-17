#!/usr/bin/env python3

'''Go through the files in the current directory and remove and caffemodel/solverstate
files where there is a higher iteration available'''

import glob,re,sys,collections,os

for suffix in ['caffemodel','solverstate']:
    files = collections.defaultdict(list)
    for fname in glob.glob('*.%s'%suffix):
        m = re.search('(.*)_iter_(\d+)\.%s'%suffix,fname)
        if m:
            prefix = m.group(1)
            i = int(m.group(2))
            files[prefix].append((i,fname))
    for (k,files) in files.items():
        toremove = sorted(files,reverse=True)[1:]
        for (i,fname) in toremove:
            print (fname)
            os.remove(fname)
