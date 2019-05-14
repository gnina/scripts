#!/usr/bin/env python3

'''Compute a single row of a distance matrix from a pdbinfo file.  
This allows for distributed processing'''

import clustering,argparse,sys
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity as fs
from rdkit.Chem.Fingerprints import FingerprintMols

def compute_ligand_similarity(smiles, pair):
    '''
    Input a list of smiles, and a pair to compute the similarity.
    Returns the indices of the pair and the similarity
    '''

    (a,b) = pair
    smi_a = smiles[a]
    mol_a = AllChem.MolFromSmiles(smi_a)
    if mol_a == None:
        mol_a = AllChem.MolFromSmiles(smi_a, sanitize=False)
    fp_a = FingerprintMols.FingerprintMol(mol_a)

    smi_b = smiles[b]
    mol_b = AllChem.MolFromSmiles(smi_b)
    if mol_b == None:
        mol_b = AllChem.MolFromSmiles(smi_b, sanitize=False)
    fp_b = FingerprintMols.FingerprintMol(mol_b)

    sim=fs(fp_a, fp_b)

    return a, b, sim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute a single row of a distance matrix and ligand similarity matrix from a pdbinfo file.')
    parser.add_argument('--pdbseqs',type=str,required=True,help="file with target names, ligand smile, and sequences (chains separated by space)")
    parser.add_argument('-r','--row',type=int,required=True,help="row to compute")
    parser.add_argument('--out',help='output file (default stdout)',type=argparse.FileType('w'),default=sys.stdout)

    
    args = parser.parse_args()
    
    target_names = []
    targets = []
    smiles = []
    for line in open(args.pdbseqs):
        toks = line.rstrip().split()
        target_names.append(toks[0])
        smiles.append(toks[1])
        targets.append(toks[2:])
        
    r = args.row
    if r < len(target_names):
        name = target_names[r]
        row = []
        for i in range(len(target_names)):
            print(target_names[i])
            (a, b, mindist) = clustering.cUTDM2(targets, (r,i))
            (la, lb, lig_sim) = compute_ligand_similarity(smiles, (r,i))
            #sanity checks
            assert a == la
            assert b == lb
            row.append((target_names[i], mindist, lig_sim))
        #output somewhat verbosely
        for (n, dist, lsim) in row:
            args.out.write('%s %s %f %f\n'%(name, n, dist, lsim))
    else:
        print("Invalid row",r,"with only",len(target_names),"targets")
