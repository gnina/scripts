from __future__ import print_function
import sys
import os
import argparse


def write_pymol_arrows(pymol_file, atoms, radius, color, scale=1.0):
    lines = []
    lines.append('run cgo_arrow.py')
    for atom in atoms:
        elem, xi, yi, zi, dx, dy, dz = atom
        xf, yf, zf = xi+scale*dx, yi+scale*dy, zi+scale*dz
        line = 'cgo_arrow [{}, {}, {}], [{}, {}, {}]' \
               .format(xi, yi, zi, xf, yf, zf)
        if radius:
            line += ', radius={}'.format(radius)
        if color:
            line += ', color={}'.format(color)
        lines.append(line)
    with open(pymol_file, 'w') as f:
        f.write('\n'.join(lines))


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


def atom_to_pdb_line(atom, idx, dosum):
    if not isinstance(idx, int) or idx < 0 or idx > 99999:
        raise TypeError('idx must be an integer from 0 to 99999 ({})'.format(idx))
    elem, x, y, z, dx, dy, dz = atom
    if len(elem) not in {1, 2}:
        raise IndexError('atom elem must be a string of length 1 or 2 ({})'.format(elem))
    if dosum:
        d = dx+dy+dz
    else:
        d = (dx**2 + dy**2 + dz**2)**0.5
    return '{:6}{:5} {:4}{:1}{:3} {:1}{:4}{:1}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6f}       {:2}{:2}' \
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


def write_pdb_file(pdb_file, atoms, dosum):
    lines = []
    for i, atom in enumerate(atoms):
        line = atom_to_pdb_line(atom, i, dosum)
        lines.append(line)
    if pdb_file:
        with open(pdb_file, 'w') as f:
            f.write('\n'.join(lines))
    else:
        print('\n'.join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description='Convert a .xyz file containing \
        atom coordinates and gradient components to a .pdb file where the b-factor \
        is the gradient magnitude')
    parser.add_argument('xyz_file')
    parser.add_argument('--sum', action='store_true', default=False,
        help='Sum gradient components instead of taking magnitude')
    parser.add_argument('-p', '--pymol_arrows', action='store_true', default=False,
        help='Output a pymol script for gradient arrows')
    parser.add_argument('-s', '--arrow_scale', type=float, default=1.0)
    parser.add_argument('-c', '--arrow_color', default=None)
    parser.add_argument('-r', '--arrow_radius', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    atoms = read_xyz_file(args.xyz_file)
    pdb_file = args.xyz_file.replace('.xyz', '.pdb')
    write_pdb_file(pdb_file, atoms, args.sum)
    if args.pymol_arrows:
        pymol_file = args.xyz_file.replace('.xyz', '.pymol')
        write_pymol_arrows(pymol_file, atoms, args.arrow_radius,
            args.arrow_color, args.arrow_scale)

