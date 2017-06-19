#!/usr/bin/env python

'''Given a pdbinfo file output sequence information for each chain'''

import clustering,argparse,sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Output sequence information for pdbs')
    parser.add_argument('--pdbfiles',type=str,required=True,help="file with target names and paths to pbdfiles of targets (separated by space)")
    parser.add_argument('--out',help='output file (default stdout)',type=argparse.FileType('w'),default=sys.stdout)

    
    args = parser.parse_args()
    
    (target_names,targets) = clustering.readPDBfiles(args.pdbfiles)
    for (name, target) in zip(target_names,targets):
        args.out.write('%s %s\n'%(name,' '.join(target)))
