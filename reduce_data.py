import os
import re
import argparse
import random


def crossval_files(prefix, numfolds):
    cvfiles = []
    for i in range(numfolds):
        trainfile = '{}train{}.types'.format(prefix, i)
        testfile = '{}test{}.types'.format(prefix, i)
        cvfiles.append((trainfile, testfile))
    return cvfiles


def reduced_file(file):
    match = re.match('(.*?)(((train|test)[0-9]+)?.types)', file)
    return match.group(1) + '_reduced' + match.group(2)


def read_lines(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    return lines


def write_reduced_lines(file, lines, factor):
    random.shuffle(lines)
    reduced = lines[:int(len(lines)/factor)]
    with open(file, 'w') as f:
        f.write(''.join(reduced))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', required=True)
    parser.add_argument('-n', '--numfolds', type=int, default=3)
    parser.add_argument('-a', '--allfolds', default=False, action='store_true')
    parser.add_argument('-f', '--factor', required=True, type=float)
    parser.add_argument('-s', '--random_seed', type=int, default=0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.random_seed)
    cvfiles = crossval_files(args.prefix, args.numfolds)
    for i, (trainfile, testfile) in enumerate(cvfiles):
        train = read_lines(trainfile)
        reduced_trainfile = reduced_file(trainfile)
        write_reduced_lines(reduced_trainfile, train, args.factor)
        print(reduced_trainfile)
        test = read_lines(testfile)
        reduced_testfile = reduced_file(testfile)
        write_reduced_lines(reduced_testfile, test, args.factor)
        print(reduced_testfile)
    if args.allfolds:
        allfile = '{}.types'.format(args.prefix)
        all = read_lines(allfile)
        reduced_allfile = reduced_file(allfile)
        write_reduced_lines(reduced_allfile, all, args.factor)
        print(reduced_allfile)

