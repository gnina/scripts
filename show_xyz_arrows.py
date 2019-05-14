#!/usr/bin/env python
'''Adds show_arrows command to pymol, which takes an xyz file'''

import sys
import os

from pymol import cmd, cgo, CmdException
from chempy import cpv


def draw_arrow(xyz1,xyz2, radius=0.5, gap=0.0, hlength=-1, hradius=-1,
              color='blue red', name=''):
    '''
Draw an arrow; borrows heavily from cgi arrows.
    '''
    radius, gap = float(radius), float(gap)
    hlength, hradius = float(hlength), float(hradius)
    xyz1 = list(xyz1)
    xyz2 = list(xyz2)
    try:
        color1, color2 = color.split()
    except:
        color1 = color2 = color
    color1 = list(cmd.get_color_tuple(color1))
    color2 = list(cmd.get_color_tuple(color2))

    normal = cpv.normalize(cpv.sub(xyz1, xyz2))

    if hlength < 0:
        hlength = radius * 3.0
    if hradius < 0:
        hradius = hlength * 0.6

    if gap:
        diff = cpv.scale(normal, gap)
        xyz1 = cpv.sub(xyz1, diff)
        xyz2 = cpv.add(xyz2, diff)

    xyz3 = cpv.add(cpv.scale(normal, hlength), xyz2)

    obj = [cgo.CYLINDER] + xyz1 + xyz3 + [radius] + color1 + color2 + \
          [cgo.CONE] + xyz3 + xyz2 + [hradius, 0.0] + color2 + color2 + \
          [1.0, 0.0]

    if not name:
        name = cmd.get_unused_name('arrow')

    cmd.load_cgo(obj, name)



def make_pymol_arrows(base, atoms, scale, color, radius):

    arrow_objs = []
    arrow_group = base + '_arrows'
    cmd.delete(arrow_group) #remove any pre-existing group
    for i, atom in enumerate(atoms):
        arrow_obj = base + '_arrow_' + str(i)
        arrow_objs.append(arrow_obj)
        elem, xi, yi, zi, dx, dy, dz = atom
        c = 1.725*radius
        xf = xi + -scale*dx + c
        yf = yi + -scale*dy + c
        zf = zi + -scale*dz + c
        draw_arrow((xi,yi,zi),(xf,yf,zf),radius=radius,color=color,name=arrow_obj)

    cmd.group(arrow_group,' '.join(arrow_objs))


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



def read_xyz_file(xyz_file):
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    n_atoms = int(lines[0])
    atoms = []
    for i in range(n_atoms):
        atom = xyz_line_to_atom(lines[2+i])
        atoms.append(atom)
    return atoms



def show_xyz_arrows(xyzfile, scale=2.0, color="white purple",radius=0.2):
    atoms = read_xyz_file(xyzfile)
    base_name = xyzfile.replace('.xyz', '')
    make_pymol_arrows(base_name, atoms, float(scale), color, float(radius))
    
    

cmd.extend('show_xyz_arrows', show_xyz_arrows)

