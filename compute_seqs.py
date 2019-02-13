#!/usr/bin/env python

'''Given a pdbinfo file output sequence information for each chain'''

import clustering,argparse,sys

def get_smiles(target_names, input_file):
	'''
	Returns a list of each of the smiles (3rd col) of input_file, indexed by target_names.
	'''

	smi_dic={}
	smi_list=[]
	with open(input_file) as filein:
		for line in filein:
			name=line.split()[0]
			smi_file=line.split()[2].rstrip()
			smi=open(smi_file).readline().split()[0]
			smi_dic[name]=smi

	for tname in target_names:
		smi_list.append(smi_dic[tname])

	return smi_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Output the needed input for compute_row. This takes the format of "<target_name> <ligand smile> <target_sequence>" separated by spaces')
    parser.add_argument('--pdbfiles',type=str,required=True,help="file with target names, paths to pbdfiles of targets, and path to smiles file of ligand (separated by space)")
    parser.add_argument('--out',help='output file (default stdout)',type=argparse.FileType('w'),default=sys.stdout)

    
    args = parser.parse_args()
    
    (target_names,targets) = clustering.readPDBfiles(args.pdbfiles)
    target_smiles = get_smiles(target_names, args.pdbfiles)

    for (name, target, smi) in zip(target_names, targets, target_smiles):
        args.out.write('%s %s %s\n'%(name, smi, ' '.join(target)))
