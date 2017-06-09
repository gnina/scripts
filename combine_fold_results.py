#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, re, argparse, sys, os
import sklearn.metrics


def read_results_file(file):
    rows = []
    with open(file, 'r') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if line:
                rows.append(map(float, line.split(' ')))
    return zip(*rows)


def write_results_file(file, *columns, **kwargs):
    mode = kwargs.get('mode', 'w')
    footer = kwargs.get('footer', '')
    with open(file, mode) as f:
        for row in zip(*columns):
            f.write(' '.join(map(str, row)) + '\n')
        if footer:
            f.write('# %s' % footer)


def last_iters_statistics(test_aucs, test_interval, last_iters):
    n_last_tests = int(last_iters/test_interval)
    last_test_aucs = [x[-n_last_tests:] for x in test_aucs]
    return np.mean(last_test_aucs), np.max(last_test_aucs), np.min(last_test_aucs)


def training_plot(plot_file, train_series, test_series):
    assert len(train_series) == len(test_series)
    fig = plt.figure()
    plt.plot(train_series, label='Train')
    plt.plot(test_series, label='Test')
    plt.legend(loc='best')
    plt.savefig(plot_file, bbox_inches='tight')


def plot_roc_curve(plot_file, fpr, tpr, auc, txt):
    assert len(fpr) == len(tpr)
    fig = plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, label='CNN (AUC=%.2f)' % auc, linewidth=4)
    plt.legend(loc='lower right',fontsize=20)
    plt.xlabel('False Positive Rate',fontsize=22)
    plt.ylabel('True Positive Rate',fontsize=22)
    plt.axes().set_aspect('equal')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.text(.05, -.25, txt, fontsize=22)
    plt.savefig(plot_file, bbox_inches='tight')


def plot_correlation(plot_file, y_aff, y_predaff, rmsd, r2):
    assert len(y_aff) == len(y_predaff)
    fig = plt.figure(figsize=(8,8))
    plt.plot(y_aff, y_predaff, 'o', label='RMSD=%.2f, R^2=%.3f (Pos)' % (rmsd, r2))
    plt.legend(loc='best', fontsize=20, numpoints=1)
    lo = np.min([np.min(y_aff), np.min(y_predaff)])
    hi = np.max([np.max(y_aff), np.max(y_predaff)])
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel('Experimental Affinity', fontsize=22)
    plt.ylabel('Predicted Affinity', fontsize=22)
    plt.axes().set_aspect('equal')
    plt.savefig(plot_file, bbox_inches='tight')


def check_file_exists(file):
    if not os.path.isfile(file):
        raise OSError('%s does not exist' % file)


def get_results_files(prefix, numfolds, affinity, two_data_sources):
    files = {}
    for i in range(numfolds):
        files[i] = {}
        files[i]['out'] = '%s.%d.out' % (prefix, i)
        files[i]['auc_finaltest'] = '%s.%d.auc.finaltest' % (prefix, i)
        if affinity:
            files[i]['rmsd_finaltest'] = '%s.%d.rmsd.finaltest' % (prefix, i)
        if two_data_sources:
            files[i]['auc_finaltest2'] = '%s.%d.auc.finaltest2' % (prefix, i)
            if affinity:
                files[i]['rmsd_finaltest2'] = '%s.%d.rmsd.finaltest2' % (prefix, i)
    for i in files:
        for file in files[i].values():
            check_file_exists(file)
    return files


def combine_fold_results(outprefix, test_interval, test_aucs, train_aucs, all_y_true, all_y_score,
                         test_rmsds, train_rmsds, all_y_aff, all_y_predaff, is_data2=False):
    two = '2' if is_data2 else ''

    #average, min, max test AUC for last 1000 iterations
    last_iters = 1000
    avg_auc, max_auc, min_auc = last_iters_statistics(test_aucs, test_interval, last_iters)
    txt = 'For the last %s iterations:\nmean AUC=%.2f  max AUC=%.2f  min AUC=%.2f' % (last_iters, avg_auc, max_auc, min_auc)

    #average aucs across folds
    mean_test_aucs = np.mean(test_aucs, axis=0)
    mean_train_aucs = np.mean(train_aucs, axis=0)

    #write test and train aucs (mean and for each fold)
    write_results_file('%s.auc.test%s' % (outprefix, two), mean_test_aucs, *test_aucs)
    write_results_file('%s.auc.train%s' % (outprefix, two), mean_train_aucs, *train_aucs)

    #training plot of mean auc across folds
    training_plot('%s_auc_train%s.pdf' % (outprefix, two), mean_train_aucs, mean_test_aucs)

    #roc curve for the last iteration - combine all tests
    if len(np.unique(all_y_true)) > 1:
        fpr, tpr, _ = sklearn.metrics.roc_curve(all_y_true, all_y_score)
        auc = sklearn.metrics.roc_auc_score(all_y_true, all_y_score)
        write_results_file('%s.auc.finaltest%s' % (outprefix, two), all_y_true, all_y_score, footer='AUC %f\n' % auc)
        plot_roc_curve('%s_roc%s.pdf' % (outprefix, two), fpr, tpr, auc, txt)

    if test_rmsds:

        #average rmsds across folds
        mean_test_rmsds = np.mean(test_rmsds, axis=0)
        mean_train_rmsds = np.mean(train_rmsds, axis=0)

        #write test and train rmsds (mean and for each fold)
        write_results_file('%s.rmsd.test%s' % (outprefix, two), mean_test_rmsds, *test_rmsds)
        write_results_file('%s.rmsd.train%s' % (outprefix, two), mean_train_rmsds, *train_rmsds)

        #training plot of mean rmsd across folds
        training_plot('%s_rmsd_train%s.pdf' % (outprefix, two), mean_train_rmsds, mean_test_rmsds)

        all_y_aff = np.array(all_y_aff)
        all_y_predaff = np.array(all_y_predaff)
        yt = np.array(all_y_true, dtype=np.bool)
        rmsdt = sklearn.metrics.mean_squared_error(all_y_aff[yt], all_y_predaff[yt])
        r2t = sklearn.metrics.r2_score(all_y_aff[yt], all_y_predaff[yt])
        write_results_file('%s.rmsd.finaltest%s' % (outprefix, two), all_y_aff, all_y_predaff, footer='RMSD,R^2 %f %f\n' % (rmsdt, r2t))

        plot_correlation('%s_rmsd%s.pdf' % (outprefix, two), all_y_aff[yt], all_y_predaff[yt], rmsdt, r2t)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Combine training results from different folds and make graphs')
    parser.add_argument('-o','--outprefix',type=str,required=True,help="Prefix for input and output files (--outprefix from train.py)")
    parser.add_argument('-n','--numfolds',type=int,required=False,help="Number of folds to combine, default is 3",default='3')
    parser.add_argument('-a','--affinity',default=False,action='store_true',required=False,help="Whether to look for affinity results files")
    parser.add_argument('-2','--two_data_sources',default=False,action='store_true',required=False,help="Whether to look for 2nd data source results files")
    parser.add_argument('-t','--test_interval',type=int,default=40,required=False,help="Number of iterations between tests")
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()

    try:
        results_files = get_results_files(args.outprefix, args.numfolds, args.affinity, args.two_data_sources)
    except OSError as e:
        print "error: %s" % e
        sys.exit(1)

    test_aucs, train_aucs, test_rmsds, train_rmsds = [], [], [], []
    all_y_true, all_y_score, all_y_aff, all_y_predaff = [], [], [], []
    test2_aucs, train2_aucs, test2_rmsds, train2_rmsds = [], [], [], []
    all_y_true2, all_y_score2, all_y_aff2, all_y_predaff2 = [], [], [], []

    #read results files
    for i in results_files:

        y_true, y_score = read_results_file(results_files[i]['auc_finaltest'])
        all_y_true.extend(y_true)
        all_y_score.extend(y_score)

        if args.affinity:
            y_aff, y_predaff = read_results_file(results_files[i]['rmsd_finaltest'])
            all_y_aff.extend(y_aff)
            all_y_predaff.extend(y_predaff)

        if args.two_data_sources:
            y_true2, y_score2 = read_results_file(results_files[i]['auc_finaltest2'])
            all_y_true2.extend(y_true2)
            all_y_score2.extend(y_score2)

            if args.affinity:
                y_aff2, y_predaff2 = read_results_file(results_files[i]['rmsd_finaltest2'])
                all_y_aff2.extend(y_aff2)
                all_y_predaff2.extend(y_predaff2)

        #test_auc train_auc train_loss lr [test_rmsd train_rmsd] [test2_auc train2_auc train2_loss [test2_rmsd train2_rmsd]]
        out_columns = read_results_file(results_files[i]['out'])

        test_auc, train_auc = out_columns[0:2]
        test_aucs.append(test_auc)
        train_aucs.append(train_auc)

        if args.affinity:
            test_rmsd, train_rmsd = out_columns[4:6]
            test_rmsds.append(test_rmsd)
            train_rmsds.append(train_rmsd)

            if args.two_data_sources:
                test2_auc, train2_auc = out_columns[6:8]
                test2_rmsd, train2_rmsd = out_columns[9:11]
                test2_aucs.append(test2_auc)
                train2_aucs.append(train2_auc)
                test2_rmsds.append(test2_rmsd)
                train2_rmsds.append(train2_rmsd)

        elif args.two_data_sources:
            test2_auc, train2_auc = out_columns[4:6]
            test2_aucs.append(test2_auc)
            train2_aucs.append(train2_auc)

    combine_fold_results(args.outprefix, args.test_interval, test_aucs, train_aucs, all_y_true, all_y_score,
                         test_rmsds, train_rmsds, all_y_aff, all_y_predaff)
    if args.two_data_sources:
        combine_fold_results(args.outprefix, args.test_interval, test2_aucs, train2_aucs, all_y_true2, all_y_score2,
                             test2_rmsds, train2_rmsds, all_y_aff2, all_y_predaff2, True)

