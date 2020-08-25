#!/usr/bin/env python3

'''
This is a script which will generate a file of commands for gnina to use cnn_minimze to generate iterative training poses.

ASSUMPTIONS
	  i) assumes all receptors are PDB files IE end in .pdb
	 ii) Assumes all docked poses or outputs from gnina will be SDF files.
	iii) The crystal ligand filenames are formatted PDBid_LignameLIGSUFFIX
	 iv) assumes file format is ROOT/POCKET/FILES
	  v) Will generate a line for every identified crystal ligand with every identified receptor in POCKET -- i.e. crossdocking.
	 vi) Assumes ligands will have the name of their corresponding crystal ligand file present in their filename. (This is especially important is using docked poses.)
	vii) Will generate  REC_LIG_lig_it#_docked.sdf files as output. (If using docked poses as well, they will have their name will have extra _it#_ parts in it, the current it# will be the leftmost one)
'''


import os, argparse, glob, re

def get_receptors(root,rec_id):
	all_pdbs=glob.glob(root+'*.pdb')
	identifier=re.compile(rec_id)
	recs=[x for x in all_pdbs if re.match(identifier,x.split('/')[-1])]
	return recs

def get_ligands(root,lig_suffix):
	all_ligs=glob.glob(root+'*'+lig_suffix)
	return all_ligs

def generate_line(receptor,ligand,outname,crystal_ligand,seed,num_modes,builtin_cnn,supplied_cnn=None,supplied_weights=None):
	if bool(supplied_cnn) and bool(supplied_weights):
		return(f'gnina -r {receptor} -l {ligand} -o {outname} --autobox_ligand {crystal_ligand} --seed {seed} --gpu --minimize --cnn_scoring refinement --num_modes {num_modes} --cnn_model {supplied_cnn} --cnn_weights {supplied_weights}\n')
	else:
		return(f'gnina -r {receptor} -l {ligand} -o {outname} --autobox_ligand {crystal_ligand} --seed {seed} --gpu --minimize --cnn_scoring refinement --num_modes {num_modes} --cnn {builtin_cnn}\n')

#grabbing the arguments
parser=argparse.ArgumentParser(description='Create cnn_minimize jobs for a dataset. Assumes dataset file structure is <ROOT>/<Identifier>/<FILES>')
parser.add_argument('-o','--outfile',type=str,required=True,help='Name for gnina job commands output file.')
parser.add_argument('-r','--root',default='./',help='ROOT for data directory structure. Defaults to current working directory.')
parser.add_argument('-ri','--rec_id',default='...._._rec.pdb',help='Regular expression to identify the receptor PDB. Defaults to ...._._rec.pdb')
parser.add_argument('-cs','--crystal_suffix',default='_lig.pdb',help='Expresssion to glob the crystal ligand PDB. Defaults to _lig.pdb. Assumes filename is PDBid_LignameLIGSUFFIX')
parser.add_argument('-ds','--docked_suffix',default='_tt_docked.sdf',help='Expression to glob docked poses. These contain the poses that need to be minimized. Default is "_tt_docked.sdf"')
parser.add_argument('-i','--iteration',type=int,required=True,help='Sets what iteration number we are doing. Adds _it#_docked.sdf to the output file for the gnina job line.')
parser.add_argument('--num_modes',type=int,default=20,help='Sets the --num_modes argument for the gnina command. Defaults to 20.')
parser.add_argument('--cnn',type=str, default='dense',help='Sets the --cnn command for the gnina command. Defaults to dense. Must be dense, general_default2018, or crossdock_default2018.')
parser.add_argument('--cnn_model',type=str,default=None,help='Override --cnn with a user provided caffe model file. If used, requires the user to pass in a weights file as well.')
parser.add_argument('--cnn_weights',type=str,default=None,help='The weights file to use with the supplied caffemodel file.')
parser.add_argument('--seed',default=42,type=int,help='Seed for the gnina commands. Defaults to 42')
parser.add_argument('--dirs',type=str,default=None,help='Supplied file containing a subset of the dataset (one pocket per line). Default behavior is to do every directory.')
args=parser.parse_args()

#double checking that the arguments are compatible
if args.cnn_model:
	assert bool(args.cnn_weights),"Didn't set cnn_weights to go with cnn_model"
else:
	assert args.cnn in set(['dense','general_default2018','crossdock_default2018']),"Must have built-in cnn be dense, general_default2018, or crossdock_default2018"
assert args.num_modes>1,"Need to set num_modes to a positive integer."
assert args.seed>0,"Need a positive seed."
assert args.iteration>0,"Need an iteration number >=1."


#now we begin.
#Step 1 -- assemble all of the directories that we will be using.
dataroot=sys.path.join(args.root,'')
todo=glob.glob(dataroot+'*/')

if args.dirs:
	subdirs=open(args.dirs).readlines()
	subdirs=[x.rstrip() for x in subdirs]
	subdirs=set(subdirs)
	todo=[x for x in todo if x.split('/')[-2] in subdirs]

#Step 2 -- main loop of the script
#set the iteration plugin variable
itname='_it'+str(args.iteration)

#	 We loop over the pockets
#TODO -- change to only do the docked poses
with open(args.outfile,'w') as outfile:
	for pocket_root in todo:
		#grab the receptors
		recs=get_receptors(pocket_root,args.rec_id)

		#grab all of the crystal ligands
		cr_ligs=get_ligands(pocket_root,args.crystal_suffix)
		
		#Grab all of the docked poses
		ligs=get_ligands(pocket_root,args.docked_suffix)
		for r in recs:
			for cl in cr_ligs:
				#determine which ligands will work -- IE which ligands have the crystal ligand indentifier in their name, and which ligands have the receptor in their name.
				lig_todo=[l for l in ligs if cl.split('/')[-1].split(args.crystal_suffix)[0] in l]
				lig_todo=[l for l in lig_todo if r.split('/')[-1].split('.pdb')[0] in l]
				for ligname in lig_todo:
					#generate the output filename
					#if args.docked_suffix and args.docked_suffix in ligname:
					outname=ligname.replace(args.docked_suffix,itname+args.docked_suffix)
					#else:
					#	rec_part=r.split('.pdb')[0]+'_'
					#	lig_part=ligname.split('/')[-1].split(args.crystal_suffix)[0]
					#	outname=rec_part+lig_part+'_lig_'+itname+'docked.sdf'

					outfile.write(generate_line(receptor=r,ligand=ligname,outname=outname,crystal_ligand=cl,seed=args.seed,num_modes=args.num_modes,builtin_cnn=args.cnn,supplied_cnn=args.cnn_model,supplied_weights=args.cnn_weights))

