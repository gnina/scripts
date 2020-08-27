#!/usr/bin/env python3
'''
This script will generate the new types file with the lines from generate_counterexample_typeslines.py

Assumptions
	i) The data structure is <ROOT>/<POCKET>/<FILES>
	ii) The name of the file containing the types lines to add is <NAME> for each pocket in the types file.
	iii) the input types file has <POCKET>/<receptor file>  from which to parse the needed pockets from.

INPUT
	i) Original types file
	ii) New types filename
	iii) Name of file in Pocket that contains the lines to add
	iv) The ROOT of the data directory

OUTPUT
	i) The new types file -- note that the lines of the new types file will not necessarily be in order.
'''

import argparse, os, re, glob

def check_exists(filename):
	if os.path.isfile(filename) and os.path.getsize(filename)>0:
		return True
	else:
		return False

parser=argparse.ArgumentParser(description='Add lines to types file and create a new one. Assumes data file structure is ROOT/POCKET/FILES.')
parser.add_argument('-i','--input',type=str,required=True,help='Types file you will be extending.')
parser.add_argument('-o','--output',type=str,required=True,help='Name of the extended types file.')
parser.add_argument('-n','--name',type=str,required=True,help='Name of the file containing the lines to add for a given pocket. This is the output of generate_counterexample_typeslines.py.')
parser.add_argument('-r','--root',default='',help='Root of the data directory. Defaults to current working directory.')
args=parser.parse_args()

completed=set()
with open(args.output,'w') as outfile:
	with open(args.input) as infile:
		for line in infile:
			outfile.write(line)
			m=re.search(r' (\S+)/',line)
			pocket=m.group(1)

			if pocket not in completed:
				completed.add(pocket)
				with open(os.path.join(args.root,pocket,args.name)) as linesfile:
					for line2 in linesfile:
						outfile.write(line2)
