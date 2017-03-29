from __future__ import print_function
import sys
import argparse
from collections import namedtuple


def xyz_line_to_atom(xyz_line):
    fields = xyz_line.split()
    elem = fields[0]
    x  = float(fields[1])
    y  = float(fields[2])
    z  = float(fields[3])
    dx = float(fields[4])
    dy = float(fields[5])
    dz = float(fields[6])
    return elem, x, y, z, dx, dy, dz


def atom_to_pdb_line(atom, idx):
    if not isinstance(idx, int) or idx < 0 or idx > 99999:
        raise TypeError('idx must be an integer from 0 to 99999 ({})'.format(idx))
    elem, x, y, z, dx, dy, dz = atom
    if len(elem) not in {1, 2}:
        raise IndexError('atom elem must be a string of length 1 or 2 ({})'.format(elem))
    d = (dx**2 + dy**2 + dz**2)**0.5
    return '{:6}{:5} {:4}{:1}{:3} {:1}{:4}{:1}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:2}{:2}' \
             .format('ATOM', idx, '', '', '', '', '', '', x, y, z, 1.0, d, elem.rjust(2), '')


def read_xyz_file(xyz_file):
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    n_atoms = int(lines[0])
    atoms = []
    for i in range(n_atoms):
        atom = xyz_line_to_atom(lines[2+i])
        atoms.append(atom)
    return atoms


def write_pdb_file(pdb_file, atoms):
    lines = []
    for i, atom in enumerate(atoms):
        line = atom_to_pdb_line(atom, i)
        lines.append(line)
    if pdb_file:
        with open(pdb_file, 'w') as f:
            f.write('\n'.join(lines))
    else:
        print('\n'.join(lines))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('xyz_file')
    parser.add_argument('-o', '--pdb_file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    atoms = read_xyz_file(args.xyz_file)
    write_pdb_file(args.pdb_file, atoms)

