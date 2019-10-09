#!/usr/bin/env python3


import sys
import os
import argparse


def write_pymol_arrows(base, structs, scale, color, radius, hradius, hlength, threshold):
    pymol_file = base + '_arrows.pymol'
    lines = []
    arrow_objs = set()
    t2 = threshold**2
    s2 = scale**2
    for i, struct in enumerate(structs):
        for j, atom in enumerate(struct):
            arrow_obj = base + '_arrow_' + str(j)
            arrow_objs.add(arrow_obj)
            elem, xi, yi, zi, dx, dy, dz = atom
            xf = xi + scale*dx
            yf = yi + scale*dy
            zf = zi + scale*dz
            line = 'cgo_arrow [{}, {}, {}], [{}, {}, {}]'.format(xi, yi, zi, xf, yf, zf)
            if len(structs) > 1:
                line += ', state={}'.format(i+1)
            if radius:
                line += ', radius={}'.format(radius)
            if hradius > 0:
                line += ', hradius={}'.format(hradius)
            if hlength > 0:
                line += ', hlength={}'.format(hlength)
            if color:
                line += ', color={}'.format(color)
            line += ', name={}'.format(arrow_obj)
            if (dx**2 + dy**2 + dz**2)*s2 > t2:
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


def read_xyz_file(xyz_file, header_len=2):

    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    structs = []
    struct_start = 0
    for i, line in enumerate(lines):
        try:
            # line index relative to struct start
            j = i - struct_start

            if j == 0 or j >= header_len + n_atoms:
                struct_start = i
                structs.append([])
                n_atoms = int(lines[i])

            elif j < header_len:
                continue

            else:
                atom = xyz_line_to_atom(lines[i])
                structs[-1].append(atom)
        except:
            print('{}:{} {}'.format(xyz_file, i, repr(line)), file=sys.stderr)
            raise

    return structs


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
    parser.add_argument('-s', '--scale', type=float, default=1.0,
        help='Arrow length scaling factor')
    parser.add_argument('-c', '--color', type=str, default='',
        help='Arrow color or pair of colors, e.g. "white black"')
    parser.add_argument('-r', '--radius', type=float, default=0.2,
        help='Radius of arrow body')
    parser.add_argument('-hr', '--hradius', type=float, default=-1,
        help='Radius of arrow head')
    parser.add_argument('-hl', '--hlength', type=float, default=-1,
        help='Length of arrow head')
    parser.add_argument('-p', '--pdb_file', action='store_true', default=False,
        help='Output a .pdb file where the b-factor is gradient magnitude')
    parser.add_argument('--sum', action='store_true', default=False,
        help='Sum gradient components instead of taking magnitude')
    parser.add_argument('-t', '--threshold', type=float, default=0,
        help="Gradient threshold for drawing arrows (using scale factor)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    structs = read_xyz_file(args.xyz_file)
    base_name = args.xyz_file.replace('.xyz', '')
    write_pymol_arrows(base_name, structs, args.scale, args.color, args.radius, args.hradius, args.hlength, args.threshold)
    if args.pdb_file:
        pdb_file = base_name + '.pdb'
        write_pdb_file(pdb_file, atoms, args.sum)

