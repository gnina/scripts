#!/usr/bin/env python3

import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import glob, struct
import sys, argparse, os

def parse_args(argv=None):
	parser=argparse.ArgumentParser(description='Script for generating the jobs that need to be run for simple visualization. Generates types files & a text file that needs to be run. This results in separating atoms along the x-axis. Can then graph the results.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-r','--recatoms',type=str,required=True,default=argparse.SUPPRESS,help='File containing Receptor atom types of your modelfile (1 per line)')
	parser.add_argument('-l','--ligatoms',type=str, required=True,default=argparse.SUPPRESS,help='File containing Ligand atom types of your modelfile (1 per line)')
	parser.add_argument('-o','--outname',type=str, default='simplegrid_predicts.txt',help='File containing commands to be evaluated to predict grid points. Note: Requires GNINASCRIPTSDIR to be set environment variable.')
	parser.add_argument('-t','--typesroot',type=str,default='simpletypes/',help='Root folder for gninatypes data generated from script.')
	parser.add_argument('-m','--model',type=str, required=True,default=argparse.SUPPRESS,help='Model file that predictions will be made with. Must end in .model')
	parser.add_argument('-w','--weights',type=str, required=True, default=argparse.SUPPRESS,help='Weights for the model file that the predictions will be made with.')
	parser.add_argument('-n','--num_points',type=int, default=200, help='Number of points. Defaults are reasonable.')
	parser.add_argument('-i','--increment',type=float, default=0.1, help='increment (Angstroms) between points. Combines with num_points multiplicitavely. Defaults for both result in 200 points spanning 20 angstroms')
	parser.add_argument('-b','--box_size',type=int,default=24, help='Size of the box. Used for sanity check that you are not trying to predict outside of box for gnina. MUST MATCH BOX OF MODEL. Defaults are default grid size for gnina')
	parser.add_argument('--plot', action='store_true',default=False, help='Flag to make 1 large plot from the data. Assumes job(s) have completed. Requires Hydrogen to be a vaild receptor. Saves pdf called simple_vis.pdf in the current working directory')
	parser.add_argument('-d','--dataroot',type=str,default='simpledata/',help='Root folder of data resulting from output of running the OUTNAME file')
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

def make_points(atom,val_range,root,mapping):#TODO -- make sure this works
	'''
	Function that makes the points needed for the types file.
	'''

	if not os.path.isdir(root+atom):
		os.mkdir(root+atom)

	counter=0
	for x in val_range:
		pos=[x,0.0,0.0]
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
	filename=root+receptor.split('/')[-1].split('_0.gnina')[0]+'_'+atom+'.types'
	with open(filename,'w') as out:
		for g in gninatypes:
			out.write('1 3.0 0.00 '+receptor+' '+g+'\n')

	return filename


if __name__=='__main__':
	args=parse_args()

	#sanitize inputs
	if not os.path.isdir(args.typesroot):
		os.mkdir(args.typesroot)

	if not os.path.isdir(args.dataroot):
		os.mkdir(args.dataroot)

	if not os.path.isdir(args.dataroot) and args.plot:
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

	mprefix=args.model.split('/')[-1].split('.model')[0]
	pattern = re.compile("^[0-9]")

	#figure out the dimensions that we are working with
	if args.num_points*args.increment > args.box_size:
		print('Error! Asking to predict for points outside of box_size')
		print('Be sure num_points*increment <= box_size')
		sys.exit()
	val_range=np.linspace(0, args.num_points*args.increment, args.num_points)

	if args.plot:
		#We need to set up the ligand groups [[Carbons],[Nitrogens],[Oxygens],[Sulfur + Phosphorous],[Fluorine, Chlorine, Bromine],[rest of them]]
		ligGroups=[
			[x for x in lig_atoms if 'Carbon' in x],
			[x for x in lig_atoms if 'Nitrogen' in x],
			[x for x in lig_atoms if 'Oxygen' in x],
			[x for x in lig_atoms if x=='Sulfur' or x=='Phosphorus'],
			[x for x in lig_atoms if x=='Fluorine' or x=='Chlorine' or x=='Bromine'],
			[x for x in lig_atoms if x!='Fluorine' and x!='Chlorine' and x!='Bromine' and x!='Sulfur' and x!='Phosphorus' and 'Oxygen' not in x and 'Nitrogen' not in x and 'Carbon' not in x]
		]

		ligcolors=[
			['seagreen','palegreen','darkturquoise','paleturquoise'],
			['mediumblue','dodgerblue','cyan'],
			['red','darkred'],
			['yellow','orange'],
			['plum','purple','magenta']
		]

		use_colors=[]
		use_groups=[]
		for list_colors,list_ligs in zip(ligcolors,ligGroups[:-1]):
			use_colors.append(list_colors[:len(list_ligs)])
			use_groups.append(list_ligs)

		if ligGroups[-1]!=[]:
			use_groups.append(ligGroups[-1])
			jet=cm=plt.get_cmap('jet')
			values=range(len(ligGroups[-1]))
			cNorm=colors.Normalize(vmin=0,vmax=values[-1])
			scalarMap=cmx.ScalarMappable(norm=cNorm,cmap=jet)
			use_colors_add=[]
			for idx in values:
				use_colors_add.append(scalarMap.to_rgba(values[idx]))

			use_colors.append(use_colors_add)

		#calculating the baseline (done with Hydrogen as a receptor)
		if 'Hydrogen' not in rec_atoms:
			print('Error!')
			print('Plot requires Hydrogen Atom types to be calculated as a receptor to be used as a baseline')
			sys.exit()

		n=len(use_groups)
		m=len(rec_atoms)
		f, axarr=plt.subplots(n,m,figsize=(50,20))
		for (j,recAtomType) in enumerate(rec_atoms):
			for (i,g) in enumerate(use_groups):
				ax=axarr[i][j]
				Xaxis=[x*args.increment for x in range(args.num_points)]

				for (lcolor,ligAtomType) in zip(use_colors[i],g):
					data=0
					databaseline=0
					fn=dataroot+recAtomType+'_rec_'+ligAtomType+'_lig_'+mprefix+'_predictscores'
					with open(fn) as f:
						lines=f.readlines()
						lines=[x.split()[0] for x in lines]
						data=np.array([float(x) for x in lines if pattern.match(x)])

					with open(dataroot+'Hydrogen_rec_'+ligAtomType+'_lig_'+mprefix+'_predictscores') as f:
						lines=f.readlines()
						lines=[x.split()[0] for x in lines]
						data_baseline=np.array([float(x) for x in lines if pattern.match(x)])

					if recAtomType=='Hydrogen':
						lol=ax.plot(Xaxis,data,label=ligAtomType.replace('XS',""),linewidth=2,color=lcolor)
					else:
						lol=ax.plot(Xaxis, data-data_baseline, label=ligAtomType.replace('XS',''),linewidth=2,color=lcolor)

				ax.add_patch(plt.Rectangle((1.2,-1),.3,2.0,facecolor='grey',alpha=.3))
				ax.add_patch(plt.Rectangle((3,-1),.5,2.0,facecolor='grey',alpha=.3))
				ax.text(1,-.95,"Covalent",rotation_mode=None,color='black',visible=True,rotation=90,verticalalignment='bottom')
				ax.text(3,-.95,"Van der Waals",color='black',visible=True,rotation=90,verticalalignment='bottom')
				ax.set_ylim(-1,1)
				
				if i==n-1:
					ax.set_xlabel("Distance From Receptor (A)",fontsize=16)
				if i==0 and recAtomType=='Hydrogen':
					ax.set_title("Ligand Baseline Score",fontsize=14)
				elif i==0:
					ax.set_title(recAtomType.replace('XS',""),fontsize=14)
				if j==0:
					ax.set_ylabel('CNN Score', fontsize=16)
				if j==m-1:
					ax.legend(bbox_to_anchor=(1,.5),loc='center left',ncol=1,fontsize=14,title="Ligand Atom Type")
				if i < n-1:
					ax.tick_params(labelbottom='off')  
				if j > 0:
					ax.tick_params(labelleft='off')
		plt.suptitle('Receptor Atom Type',fontsize=24)
		plt.subplots_adjust(wspace=0.05,hspace=0.05,top=0.94)
		plt.savefig('simple_vis.pdf')

	else:
		with open(args.outname,'w') as outfile:
			#make each gninatypes file
			for atom in todo:
				make_points(atom,val_range,types_root, inv_map[atom])

			#make types file for each cross-product atoms
			for r_atom in rec_atoms:
				rec = types_root+r_atom+'/'+r_atom+'_0.gninatypes'
				for l_atom in lig_atoms:
					#make types file
					tn = make_types(l_atom, types_root, rec)
					newname = dataroot+r_atom+'_rec_'+l_atom+'_lig_'+mprefix+'_predictscores\n'

					#write line corresponding to that types file in outfile
					outfile.write('$GNINASCRIPTSDIR/predict.py -m '+args.model+' -w '+args.weights+' -i '+tn+' --rotation 100 > '+newname)
