#!/usr/bin/env python3

'''convert gninatypes file to xyz file'''
import struct, sys, argparse
from functools import partial
import molgrid

names = molgrid.GninaIndexTyper().get_type_names()

def elem(t):
    '''convert type index into element string'''
    name = names[t]
    if 'Hydrogen' in name:
        return 'H'
    elif 'Carbon' in name:
        return 'C'
    elif 'Nitrogen' in name:
        return 'N'
    elif 'Oxygen' in name:
        return 'O'
    elif 'Sulfur' in name:
        return 'S'
    elif 'Phosphorus' == name:
        return 'P'
    elif 'Fluorine' == name:
        return 'F'
    elif 'Chlorine' == name:
        return 'Cl'
    elif 'Bromine' == name:
        return 'Br'
    elif 'Iodine' == name:
        return 'I'
    elif 'Magnesium' == name:
        return 'Mg'
    elif 'Manganese' == name:
        return 'Mn'
    elif 'Zinc' == name:
        return 'Zn'
    elif 'Calcium' == name:
        return 'Ca'
    elif 'Iron' == name:
        return 'Fe'
    elif 'Boron' == name:
        return 'B'
    else:
        return 'X'
        

parser = argparse.ArgumentParser()
parser.add_argument('input',type=str,help='gninatypes file')
parser.add_argument('output', default='-',nargs='?',type=argparse.FileType('w'),help='output xyz')

args = parser.parse_args()


struct_fmt = 'fffi'
struct_len = struct.calcsize(struct_fmt)
struct_unpack = struct.Struct(struct_fmt).unpack_from

with open(args.input,'rb') as tfile:
        results = [struct_unpack(chunk) for chunk in iter(partial(tfile.read, struct_len), b'')]

args.output.write('%d\n'%len(results)) # number atoms
args.output.write(args.input+'\n') #comment
for x,y,z,t in results:
    args.output.write('%s\t%f\t%f\t%f\n'%(elem(t),x,y,z))

