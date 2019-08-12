#!/usr/bin/env python3


import sys
import os
import argparse


def write_pymol_arrows(base, atoms, scale, color, radius, threshold):
    pymol_file = base + '_arrows.pymol'
    lines = []
    lines.append('run cgo_arrow.py')
    arrow_objs = []
    t2 = threshold**2
    s2 = scale**2
    for i, atom in enumerate(atoms):
        arrow_obj = base + '_arrow_' + str(i)
        arrow_objs.append(arrow_obj)
        elem, xi, yi, zi, dx, dy, dz = atom
        c = 1.725*radius
        xf = xi + -scale*dx + c
        yf = yi + -scale*dy + c
        zf = zi + -scale*dz + c
        line = 'cgo_arrow [{}, {}, {}], [{}, {}, {}]'.format(xi, yi, zi, xf, yf, zf)
        if radius:
            line += ', radius={}'.format(radius)
        if color:
            line += ', color={}'.format(color)
        line += ', name={}'.format(arrow_obj)
        if( (dx**2 + dy**2 + dz**2)*s2 > t2 ): # Check threshold
            lines.append(line)
    arrow_group = base + '_arrows'
    line = 'group {}, {}'.format(arrow_group, ' '.join(arrow_objs))
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
    parser = argparse.ArgumentParser(description='Output a pymol script that creates \
        arrows from an .xyz file containing atom coordinates and gradient components, \
        can also create a .pdb file where the b-factor is the gradient magnitude')
    parser.add_argument('xyz_file')
    parser.add_argument('-s', '--scale', type=float, default=1.0)
    parser.add_argument('-c', '--color', type=str, default='black green')
    parser.add_argument('-r', '--radius', type=float, default=0.2)
    parser.add_argument('-p', '--pdb_file', action='store_true', default=False,
        help='Output a .pdb file where the b-factor is gradient magnitude')
    parser.add_argument('--sum', action='store_true', default=False,
        help='Sum gradient components instead of taking magnitude')
    parser.add_argument('-t', '--threshold', type=float, default=0,
        help="Lower thrashold for arrow length (scaled)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    atoms = read_xyz_file(args.xyz_file)
    base_name = args.xyz_file.replace('.xyz', '')
    write_pymol_arrows(base_name, atoms, args.scale, args.color, args.radius, args.threshold)
    if args.pdb_file:
        pdb_file = base_name + '.pdb'
        write_pdb_file(pdb_file, atoms, args.sum)

