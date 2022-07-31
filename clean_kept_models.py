#!/usr/bin/env python3

'''Go through the files in the current directory and remove and caffemodel/solverstate
files where there is a higher iteration available'''

import glob,re,sys,collections,os

prefixes = sys.argv[1:]
if not prefixes:
    prefixes = ['.']

for dirname in prefixes:
    for suffix in ['caffemodel','solverstate','checkpoint','gen_model_state','gen_solver_state']:
        files = collections.defaultdict(list)
        for fname in glob.glob('%s/*.%s'%(dirname,suffix)):
            m = re.search('(.*)_iter_(\d+)\.%s'%suffix,fname)
            if m:
                prefix = m.group(1)
                i = int(m.group(2))
                files[prefix].append((i,fname))
        for (k,files) in list(files.items()):
            toremove = sorted(files,reverse=True)[1:]
            for (i,fname) in toremove:
                print (fname)
                os.remove(fname)
