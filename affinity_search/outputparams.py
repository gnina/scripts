#!/usr/bin/env python

'''
Output the parameters that makemodel supports with their ranges
'''

import makemodel, argparse

#extract from arguments to makemodel
opts = makemodel.getoptions()
for (name,vals) in sorted(opts.items()):
    print name,vals
