#!/usr/bin/env python

'''
Output the parameters that makemodel supports with their ranges
'''

import makemodel
import json, sys
from collections import OrderedDict

#extract from arguments to makemodel
opts = makemodel.getoptions()

create = 'CREATE TABLE params (rmse DOUBLE, top DOUBLE, R DOUBLE, auc DOUBLE'

#everything else make a string
for (name,vals) in sorted(opts.items()):
	create += ', %s VARCHAR(32)' % name

create += ');'

print(create)
