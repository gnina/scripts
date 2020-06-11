#!/usr/bin/ python

import numpy as np
import re
import gridData, glob, struct
import sys, argparse, os, subprocess

def parse_args(argv=None):
	parser=argparse.ArgumentParser(description='Script for generating the jobs that need to be run for visualization. Generates types files & a text file that needs to be run. Can make a DX file for visualization', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-r','--recatoms',type=str,required=True,default=argparse.SUPPRESS,help='File containing Receptor atom types of your modelfile (1 per line)\n')
	parser.add_argument('-l','--ligatoms',type=str, required=True,default=argparse.SUPPRESS,help='File containing Ligand atom types of your modelfile (1 per line)\n')
	parser.add_argument('-o','--outname',type=str, default='grid_predicts.txt',help='File containing commands to be evaluated to predict grid points. Note: Requires GNINASCRIPTSDIR to be a set environment variable.\n')
	parser.add_argument('-t','--typesroot',type=str,default='types/',help='Root folder for gninatypes data generated from script.\n')
	parser.add_argument('-m','--model',type=str, required=True,default=argparse.SUPPRESS,help='Model file that predictions will be made with. Must end in .model\n')
	parser.add_argument('-w','--weights',type=str, required=True, default=argparse.SUPPRESS,help='Weights for the model file that the predictions will be made with.\n')
	parser.add_argument('-p','--test_pdb',type=str,default='gly_gly_gly.pdb', help='pdbfile of receptor, centered at the origin for visualization\n')
	parser.add_argument('-c','--cube_length', type=float, default=24.0, help='Width of cube for grid box of points. Defaults are reasonable\n')
	parser.add_argument('-n','--num_points',type=int, default=20, help='Number of points per half of the box (ex 20 means there will be 39x39x39 points total). Defaults are reasonable.\n')
	parser.add_argument('--make_dx', action='store_true',default=False, help='Flag to make dx files from the data. Assumes job(s) have completed.\n')
	parser.add_argument('-d','--dataroot',type=str,default='data/',help='Root folder of data resulting from output\n')
	args=parser.parse_args(argv)

	return args

def path_checker(filename):
	if os.path.isfile(filename) and os.path.getsize(filename)>0:
		return True
	else:
		return False


def get_atoms(filename):
	'''
	Function that reads the atom types from filename & returns them as a list
	'''
	listo=[]
	with open(filename) as infile:
		for line in infile:
			item=line.rstrip()
			listo.append(item)

	return listo

def make_points(atom,val_range,root,mapping):
	'''
	Function that makes the points needed for the types file.
	'''

	if not os.path.isdir(root+atom):
		os.mkdir(root+atom)

	counter=0
	for x in val_range:
		for y in val_range:
			for z in val_range:
				pos=[x,y,z]
				pos=struct.pack('f'*len(pos),*pos)
				identity=[mapping]
				identity=struct.pack('i'*len(identity),*identity)
				with open(root+atom+'/'+atom+'_'+str(counter)+'.gninatypes','wb') as f:
					f.write(pos)
					f.write(identity)
				counter+=1

def make_types(atom, root, receptor):
	'''
	Function that writes a types file for all the points created from make_points in root

	Returns the name of the file
	'''
	def atoi(text):
		return int(text) if text.isdigit() else text

	def natural_keys(text):
		return [ atoi(c) for c in re.split(r'(\d+)', text) ]

	gninatypes=glob.glob(root+atom+'/'+atom+'*.gninatypes')
	gninatypes.sort(key=natural_keys)
	filename=root+receptor.split('_0.gnina')[0]+'_'+atom+'.types'
	with open(filename,'w') as out:
		for g in gninatypes:
			out.write('1 3.0 0.00 '+receptor+' '+g+'\n')

	return filename

def make_dx(filename, num_on_axis, min_point, val_delta):
	'''
	Function that takes the filename IE output of jobs, and makes a dx file from the results for visualization
	'''

	with open(filename) as fin:
		data=fin.readlines()

        if len(data) == 0:
            return None,None
	l=filename.split('_predictscores')[0]

	pattern=re.compile("^[0-9]")
	data=[float(x.split()[0]) for x in data if pattern.match(x)]
	scores=np.array(data)
	dxdata=scores.reshape(num_on_axis,num_on_axis,num_on_axis)
	test=dxdata.round(4)
	g=gridData.Grid(dxdata,origin=min_point, delta=val_delta)
	g.export(l+"grid","DX")
	return dxdata,test

def gninatyper(pdbfilename):
	'''
	Function that takes in a pdbfile and converts it to a gninatypes file via gninatyper

	Returns 1 on failed gninatyper
	Returns newfilename on success.
	'''

	newname=pdbfilename.split('.')[0]

	try:
		subprocess.call('gninatyper '+pdbfilename+' '+newname,shell=True)
	except:
		return 1

	return newname+'_0.gninatypes'

if __name__=='__main__':
	args=parse_args()

	#perform arguments check to terminate early?

	#sanitize inputs
	if not os.path.isdir(args.typesroot):
		os.mkdir(args.typesroot)

	if not os.path.isdir(args.dataroot):
		os.mkdir(args.dataroot)

	if not os.path.isdir(args.dataroot) and args.make_dx:
		print('Error! Specified plotting, but the dataroot does not exist!')
		print('Could not find the directory: '+args.dataroot)
		sys.exit()

	if not path_checker(args.recatoms) or not path_checker(args.ligatoms):
		print('Error!')
		print('Could not locate either: '+args.recatoms+' or '+args.ligatoms)
		sys.exit()

	if not path_checker(args.model):
		print('Error!')
		print('Could not locate: '+args.model)
		sys.exit()

	if not path_checker(args.weights):
		print('Error!')
		print('Could not locate: '+args.weights)
		sys.exit()

	if not path_checker(args.test_pdb):
		print('Error!')
		print('Could not locate: '+args.test_pdb)
		sys.exit()

	#Now we are ready to start the program!

	#making atom mapping -- BLAH hardcoded. Not sure if this is changing, but is critical to functionality
	inv_map = {
		'Hydrogen':0,
		'PolarHydrogen':1,
		'AliphaticCarbonXSHydrophobe':2 ,
		'AliphaticCarbonXSNonHydrophobe':3 ,
		'AromaticCarbonXSHydrophobe':4 ,
		'AromaticCarbonXSNonHydrophobe':5 ,
		'Nitrogen':6,
		'NitrogenXSDonor':7,
		'NitrogenXSDonorAcceptor':8,
		'NitrogenXSAcceptor':9,
		'Oxygen':10,
		'OxygenXSDonor':11,
		'OxygenXSDonorAcceptor':12,
		'OxygenXSAcceptor':13,
		'Sulfur':14,
		'SulfurAcceptor':15,
		'Phosphorus':16,
		'Fluorine':17,
		'Chlorine':18,
		'Bromine':19,
		'Iodine':20,
		'Magnesium':21,
		'Manganese':22,
		'Zinc':23,
		'Calcium':24,
		'Iron':25,
		'GenericMetal':26,
		'Boron':27,
	}

	#now we need to figure out which atom types we are working with
	lig_atoms=get_atoms(args.ligatoms)
	rec_atoms=get_atoms(args.recatoms)
	todo=list(set(lig_atoms+rec_atoms))

	#making sure that the roots are formatted appropriately
	types_root=args.typesroot
	if types_root[-1]!='/':
		types_root+='/'
	dataroot=args.dataroot
	if dataroot[-1]!='/':
		dataroot+='/'

	prefix=args.test_pdb.split('/')[-1].split('.pdb')[0]
	mprefix=args.model.split('/')[-1].split('.model')[0]

	#figure out the dimensions that we are working with
	rad=args.cube_length/2.0
	testpos=np.linspace(0,rad,args.num_points)
	testneg=np.linspace(-1*rad,0,args.num_points)
	val_range=list(testneg[:-1])+list(testpos)
	num_on_axis=len(val_range)
	minimum_point=(-1*rad, -1*rad, -1*rad)
	val_delta=val_range[1]-val_range[0]

	#The bulk of the script
	if args.make_dx:
		for atom in todo:
			print('Working on '+atom)
			#make the dx file
			data_name=dataroot+prefix+'_rec_'+atom+'_lig_'+mprefix+'_predictscores'
			_,_ = make_dx(data_name, num_on_axis, minimum_point, val_delta)
			print('Made dx file in: '+dataroot)
	else:
		with open(args.outname,'w') as outfile:
			for atom in todo:
				print('Working on '+atom)

				#make the points
				make_points(atom, val_range, types_root, inv_map[atom])
				print('Made points in: '+types_root+atom)

				#make the gninatypes file
				gninatypes_filename=gninatyper(args.test_pdb)
				if gninatypes_filename==1:
					print('Error with gninatyper!')
					sys.exit()

				if not path_checker(gninatypes_filename):
					print('Error!')
					print(gninatypes_filename+' is an empty file!')
					sys.exit()

				#then make the files
				working_name=make_types(atom, types_root, gninatypes_filename)
				print('Made typesfile in: '+types_root)

				#and write the newline
				outfile.write('$GNINASCRIPTSDIR/predict.py -m '+args.model+' -w '+args.weights+' -i '+working_name+' --rotation 100 > '+dataroot+prefix+'_rec_'+atom+'_lig_'+mprefix+'_predictscores\n')
