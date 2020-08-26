#!/usr/bin/env python3
'''
This script exists to generate the unique ligand sdf files in a given pocket, by grabbing all of the existing docked poses
and then calculating which of them are less than a certain RMSD threshold between one another.

Assumptions:
	i) The crystal ligand files are named <PDBid>_<ligand name><CRYSTAL SUFFIX>
	ii) the directory structure of the data is <ROOT>/<POCKET>/<FILES>
	iii) you have obrms installed and accessible from the commandline

Input:
	i) the Pocket directory you are working on
	ii) the root of the pocket directories
	iii) the suffix for the docked poses
	iv) the suffix for the crystal poses
	v) the desired suffix for the "unique pose sdf"
	vi) the threshold RMSD to determine unique poses

Output:
	i) a file for each ligand in the pocket containing the unique poses for that ligand.
'''

import argparse, subprocess, glob, re, os
from rdkit.Chem import AllChem as Chem
import pandas as pd

def check_exists(filename):
	if os.path.isfile(filename) and os.path.getsize(filename)>0:
		return True
	else:
		return False

def run_obrms_cross(filename):
	'''
	This function returns a pandas dataframe of the RMSD between every pose and every other pose, which is generated using obrms -x
	'''

	csv=subprocess.check_output('obrms -x '+filename,shell=True)
	csv=str(csv,'utf-8').rstrip().split('\n')
	data=pd.DataFrame([x.split(',')[1:] for x in csv],dtype=float)
	return data

parser=argparse.ArgumentParser(description='Create ligname<OUTSUFFIX> files for use with generate_counterexample_typeslines.py.')
parser.add_argument('-p','--pocket',type=str,required=True,help='Name of the pocket that you will be generating the file for.')
parser.add_argument('-r','--root',type=str,required=True,help='PATH to the ROOT of the pockets.')
parser.add_argument('-ds','--docked_suffix',default='_tt_docked.sdf', help='Expression to glob docked poses. These contain the poses that need to be uniqified. Default is "_tt_docked.sdf"')
parser.add_argument('-cs','--crystal_suffix',default='_lig.pdb', help='Expression to glob the crystal ligands. Default is "_lig.pdb"')
parser.add_argument('-os','--out_suffix',required=True,help='End of the filename for LIGNAME<OUTSUFFIX>. This will be the --old_unique_suffix for generate_counterexample_typeslines.py.')
parser.add_argument('--unique_threshold',default=0.25,help='RMSD threshold for unique poses. IE poses with RMSD > thresh are considered unique. Defaults to 0.25.')
args=parser.parse_args()

assert args.unique_threshold >0, "Unique RMSD threshold needs to be positive"

#setting the myroot variable
myroot=os.path.join(args.root,args.pocket,'')


#1) gather the crystal files & pull out the crystal names present in the pocket
crystal_files=glob.glob(myroot+'*'+args.crystal_suffix)
crystal_names=set([x.split('/')[-1].split(args.crystal_suffix)[0].split('_')[1] for x in crystal_files])

#2) main loop
for cr_name in crystal_names:
	if cr_name!='iqz':
		continue
	print(cr_name)
	#i) grab all of the docked files
	docked_files=glob.glob(myroot+'*_'+cr_name+'_*'+args.docked_suffix)
	print(docked_files)

	#ii) make sure that the "working sdf file" does not exist
	sdf_name=myroot+'___.sdf'
	if check_exists(sdf_name):
		os.remove(sdf_name)

	#iii) write all of the previously docked poses into the 'working sdf file'
	w=Chem.SDWriter(sdf_name)
	for file in docked_files:
		supply=Chem.SDMolSupplier(file,sanitize=False)
		for mol in supply:
			w.write(mol)
	w=None

	#iv) run obrms cross to calculate the RMSD between every pair of poses
	unique_data=run_obrms_cross(sdf_name)

	#v) determine the "unique poses"
	assignments={}
	for (r,row) in unique_data.iterrows():
		if r not in assignments:
			for simi in row[row<args.unique_threshold].index:
				if simi not in assignments:
					assignments[simi]=r

	to_remove=set([k for (k,v) in assignments.items() if k!=v])
	
	#vi)  write the unique files to the sdf
	new_unique_sdfname=myroot+cr_name+args.out_suffix
	w=Chem.SDWriter(new_unique_sdfname)
	supply=Chem.SDMolSupplier(sdf_name,sanitize=False)
	for i,mol in enumerate(supply):
		if i not in to_remove:
			w.write(mol)