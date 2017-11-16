#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, re, argparse, sys, os
import sklearn.metrics


def read_output_file(file, header=False):
    '''Read columns of float data from a file, ignoring # comments'''
    rows = []
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if header and i == 0:
                continue
            line = line.split('#', 1)[0].strip()
            if line:
                rows.append(map(float, line.split(' ')))
    return zip(*rows)


def write_output_file(file, *columns, **kwargs):
    '''Write columns of data to a file, with optional footer comment'''
    footer = kwargs.get('footer', '')
    with open(file, 'w') as f:
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


def plot_correlation(plot_file, aff_true, aff_pred, rmsd, r2):
    assert len(aff_true) == len(aff_pred)
    fig = plt.figure(figsize=(8,8))
    plt.plot(aff_true, aff_pred, 'o', label='RMSE=%.2f, R^2=%.3f (Pos)' % (rmsd, r2))
    plt.legend(loc='best', fontsize=20, numpoints=1)
    lo = np.min([np.min(aff_true), np.min(aff_pred)])
    hi = np.max([np.max(aff_true), np.max(aff_pred)])
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel('Experimental Affinity', fontsize=22)
    plt.ylabel('Predicted Affinity', fontsize=22)
    plt.axes().set_aspect('equal')
    plt.savefig(plot_file, bbox_inches='tight')


def check_file_exists(file):
    if not os.path.isfile(file):
        raise OSError('%s does not exist' % file)


def get_fold_files(prefix, foldnums, scoring, affinity, rmsd, two_data_sources):
    files = {}
    if foldnums is None:
        foldnums = set()
        pattern = r'%s\.(\d+)\.(training_output|final_(train|test)2?_(score|aff|rmsd))$' % prefix
        for file in glob.glob(prefix + '*'):
            match = re.match(pattern, file)
            if match:
                foldnums.add(int(match.group(1)))
    elif isinstance(foldnums, str):
        foldnums = [int(i) for i in foldnums.split(',') if i]
    outputs = []
    if scoring:
        outputs.append('score')
    if affinity:
        outputs.append('aff')
    if rmsd:
        outputs.append('rmsd')
    if two_data_sources:
        data_nums = ['', '2']
    else:
        data_nums = ['']
    for fold in foldnums:
        files[fold] = {}
        files[fold]['out'] = '%s.%d.training_output' % (prefix, fold)
        for part in ['test', 'train']:
            for num in data_nums:
                files[fold][part+num] = {}
                for output in outputs:
                    fold_file = '%s.%d.final_%s%s_%s' % (prefix, fold, part, num, output)
                    files[fold][part+num][output] = fold_file
                    check_file_exists(fold_file)
    return files


def filter_actives(values, y_true):
    return np.array(values)[np.array(y_true, dtype=np.bool)]


def combine_fold_output(test_metrics, train_metrics, test_labels, test_preds, train_labels, train_preds,
                         outprefix, test_interval, affinity=False, rmsd=False, second_data_source=False,
                         filter_actives_test=None, filter_actives_train=None):
    '''Make results files and graphs combined from results for
    separate crossvalidation folds. test_metrics and train_metrics
    are lists of lists of AUCs or RMSEs for each fold, for each
    test_interval. test_labels and test_preds are labels and final
    test predictions for each test fold, in a single list. train_labels
    and train_preds are labels and predictions for each train fold in
    a single list. affinity says whether to interpret the labels and
    predictions as affinities, and the metric as RMSE instead of AUC.
    rmsd says to interpret them as RMSD and RMSE.
    second_data_source sets the output file names to reflect whether
    the results are for the second data source of a combined data model.'''

    if affinity:
        output_name = 'aff'
        metric_name = 'aff_rmse'
    elif rmsd:
        output_name = 'rmsd'
        metric_name = 'rmsd_rmse'
    else:
        output_name = 'score'
        metric_name = 'auc'

    if second_data_source:
        num = '2'
    else:
        num = ''

    #average metric across folds
    mean_test_metrics = np.mean(test_metrics, axis=0)
    mean_train_metrics = np.mean(train_metrics, axis=0)

    #write test and train metrics (mean and for each fold)
    metrics_file = '%s.test%s_%s' % (outprefix, num, metric_name)
    write_output_file(metrics_file, mean_test_metrics, *test_metrics)

    metrics_file = '%s.train%s_%s' % (outprefix, num, metric_name)
    write_output_file(metrics_file, mean_train_metrics, *train_metrics)

    #training plot of mean test and train metric across folds
    training_file = '%s.%s_training%s.pdf' % (outprefix, metric_name, num)
    training_plot(training_file, mean_train_metrics, mean_test_metrics)

    if affinity or rmsd:

        if filter_actives_test:
            test_preds = filter_actives(test_preds, filter_actives_test)
            test_labels = filter_actives(test_labels, filter_actives_test)

        #correlation plots for last test iteration
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(test_preds, test_labels))
        r2 = sklearn.metrics.r2_score(test_preds, test_labels)

        final_output_file = '%s.final_test%s_%s' % (outprefix, num, output_name)
        write_output_file(final_output_file, test_preds, test_labels, footer='RMSE,R^2 %f %f\n' % (rmse, r2))

        corr_file = '%s.test%s_%s_corr.pdf' % (outprefix, num, output_name)
        plot_correlation(corr_file, test_preds, test_labels, rmse, r2)

        if filter_actives_train:
            train_preds = filter_actives(train_preds, filter_actives_train)
            train_labels = filter_actives(train_labels, filter_actives_train)

        rmsd = np.sqrt(sklearn.metrics.mean_squared_error(train_preds, train_labels))
        r2 = sklearn.metrics.r2_score(train_preds, train_labels)

        final_output_file = '%s.final_train%s_%s' % (outprefix, num, output_name)
        write_output_file(final_output_file, train_preds, train_labels, footer='RMSE,R^2 %f %f\n' % (rmse, r2))

        corr_file = '%s.train%s_%s_corr.pdf' % (outprefix, num, output_name)
        plot_correlation(corr_file, train_preds, train_labels, rmse, r2)

    else: # binary classification

        #roc curves for the last test iteration
        if len(np.unique(test_labels)) > 1:

            last_iters = 1000
            avg_auc, max_auc, min_auc = last_iters_statistics(test_metrics, test_interval, last_iters)
            txt = 'For the last %s iterations:\nmean AUC=%.2f  max AUC=%.2f  min AUC=%.2f' % (last_iters, avg_auc, max_auc, min_auc)

            fpr, tpr, _ = sklearn.metrics.roc_curve(test_labels, test_preds)
            auc = sklearn.metrics.roc_auc_score(test_labels, test_preds)

            final_output_file = '%s.final_test%s_%s' % (outprefix, num, output_name)
            write_output_file(final_output_file, test_labels, test_preds, footer='AUC %f\n' % auc)

            roc_file = '%s.test%s_roc.pdf' % (outprefix, num)
            plot_roc_curve(roc_file, fpr, tpr, auc, txt)

        if len(np.unique(train_labels)) > 1:

            last_iters = 1000
            avg_auc, max_auc, min_auc = last_iters_statistics(train_metrics, test_interval, last_iters)
            txt = 'For the last %s iterations:\nmean AUC=%.2f  max AUC=%.2f  min AUC=%.2f' % (last_iters, avg_auc, max_auc, min_auc)

            fpr, tpr, _ = sklearn.metrics.roc_curve(train_labels, train_preds)
            auc = sklearn.metrics.roc_auc_score(train_labels, train_preds)

            final_output_file = '%s.final_train%s_%s' % (outprefix, num, output_name)
            write_output_file(final_output_file, train_labels, train_preds, footer='AUC %f\n' % auc)

            roc_file = '%s.train%s_roc.pdf' % (outprefix, num)
            plot_roc_curve(roc_file, fpr, tpr, auc, txt)



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Combine training results from different folds and make graphs')
    parser.add_argument('-o','--outprefix',type=str,required=True,help="Prefix for input and output files (--outprefix from train.py)")
    parser.add_argument('-n','--foldnums',type=str,required=False,help="Fold numbers to combine, default is to determine using glob",default=None)
    parser.add_argument('-s','--scoring',default=False,action='store_true',required=False,help="Look for binary classification/scoring files")
    parser.add_argument('-a','--affinity',default=False,action='store_true',required=False,help="Look for affinity prediction files")
    parser.add_argument('-r','--rmsd',default=False,action='store_true',required=False,help="Look for RMSD prediction files")
    parser.add_argument('-2','--two_data_sources',default=False,action='store_true',required=False,help="Whether to look for 2nd data source results files")
    parser.add_argument('-t','--test_interval',type=int,default=40,required=False,help="Number of iterations between tests")
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()

    if not (args.scoring or args.affinity or args.rmsd):
        print 'error: must look for --scoring, --affinity, or --rmsd output files'
        sys.exit(1)

    try:
        fold_files = get_fold_files(args.outprefix, args.foldnums, args.scoring, args.affinity, args.rmsd, args.two_data_sources)
    except OSError as e:
        print "error: %s" % e
        sys.exit(1)

    if len(fold_files) == 0:
        print "error: missing fold output files"
        sys.exit(1)

    for i in fold_files:
        for key in sorted(fold_files[i], key=len):
            print str(i).rjust(3), key.rjust(15), fold_files[i][key]

    test_auc, train_auc = [], []
    test_y_true, train_y_true = [], []
    test_y_score, train_y_score = [], []

    test_aff_rmse, train_aff_rmse = [], []
    test_aff_true, train_aff_true = [], []
    test_aff_pred, train_aff_pred = [], []

    test_rmsd_rmse, train_rmsd_rmse = [], []
    test_rmsd_true, train_rmsd_true = [], []
    test_rmsd_pred, train_rmsd_pred = [], []

    test2_auc, train2_auc = [], []
    test2_y_true, train2_y_true = [], []
    test2_y_score, train2_y_score = [], []

    test2_aff_rmse, train2_aff_rmse = [], []
    test2_aff_true, train2_aff_true = [], []
    test2_aff_pred, train2_aff_pred = [], []

    test2_rmsd_rmse, train2_rmsd_rmse = [], []
    test2_rmsd_true, train2_rmsd_true = [], []
    test2_rmsd_pred, train2_rmsd_pred = [], []

    #read results files
    for i in fold_files:

        if args.scoring:

            y_true, y_score = read_output_file(fold_files[i]['test']['score'])
            test_y_true.extend(y_true)
            test_y_score.extend(y_score)

            y_true, y_score = read_output_file(fold_files[i]['train']['score'])
            train_y_true.extend(y_true)
            train_y_score.extend(y_score)

        if args.affinity:

            aff_true, aff_pred = read_output_file(fold_files[i]['test']['aff'])
            test_aff_true.extend(aff_true)
            test_aff_pred.extend(aff_pred)

            aff_true, aff_pred = read_output_file(fold_files[i]['train']['aff'])
            train_aff_true.extend(aff_true)
            train_aff_pred.extend(aff_pred)

        if args.rmsd:

            rmsd_true, rmsd_pred = read_output_file(fold_files[i]['test']['rmsd'])
            test_rmsd_true.extend(rmsd_true)
            test_rmsd_pred.extend(rmsd_pred)

            rmsd_true, rmsd_pred = read_output_file(fold_files[i]['train']['rmsd'])
            train_rmsd_true.extend(rmsd_true)
            train_rmsd_pred.extend(rmsd_pred)

        if args.two_data_sources:

            if args.scoring:

                y_true, y_score = read_output_file(fold_files[i]['test2']['score'])
                test2_y_true.extend(y_true)
                test2_y_score.extend(y_score)

                y_true, y_score = read_output_file(fold_files[i]['train2']['score'])
                train2_y_true.extend(y_true)
                train2_y_score.extend(y_score)

            if args.affinity:

                aff_true, aff_pred = read_output_file(fold_files[i]['test2']['aff'])
                test2_aff_true.extend(aff_true)
                test2_aff_pred.extend(aff_pred)

                aff_true, aff_pred = read_output_file(fold_files[i]['train2']['aff'])
                train2_aff_true.extend(aff_true)
                train2_aff_pred.extend(aff_pred)

            if args.rmsd:

                rmsd_true, rmsd_pred = read_output_file(fold_files[i]['test2']['rmsd'])
                test2_rmsd_true.extend(rmsd_true)
                test2_rmsd_pred.extend(rmsd_pred)

                rmsd_true, rmsd_pred = read_output_file(fold_files[i]['train2']['rmsd'])
                train2_rmsd_true.extend(rmsd_true)
                train2_rmsd_pred.extend(rmsd_pred)

        metrics = read_output_file(fold_files[i]['out'], header=True)
        col = 0 #ignore loss for now
        if args.scoring:
            test_auc.append(metrics[col])
            col += 2
        if args.affinity:
            test_aff_rmse.append(metrics[col])
            col += 2
        if args.rmsd:
            test_rmsd_rmse.append(metrics[col])
            col += 2
        if args.scoring:
            train_auc.append(metrics[col])
            col += 2
        if args.affinity:
            train_aff_rmse.append(metrics[col])
            col += 2
        if args.rmsd:
            train_rmsd_rmse.append(metrics[col])
            col += 2
        if args.two_data_sources:
            if args.scoring:
                test2_auc.append(metrics[col])
                col += 2
            if args.affinity:
                test2_aff_rmse.append(metrics[col])
                col += 2
            if args.rmsd:
                test2_rmsd_rmse.append(metrics[col])
                col += 2
            if args.scoring:
                train2_auc.append(metrics[col])
                col += 2
            if args.affinity:
                train2_aff_rmse.append(metrics[col])
                col += 2
            if args.rmsd:
                train2_rmsd_rmse.append(metrics[col])
                col += 2

    if args.scoring:
        combine_fold_output(test_auc, train_auc, test_y_true, test_y_score, train_y_true, train_y_score,
                            args.outprefix, args.test_interval, affinity=False, second_data_source=False)

    if args.affinity:
        combine_fold_output(test_aff_rmse, train_aff_rmse, test_aff_true, test_aff_pred, train_aff_true, train_aff_pred,
                            args.outprefix, args.test_interval, affinity=True, second_data_source=False,
                            filter_actives_test=test_y_true, filter_actives_train=train_y_true)

    if args.rmsd:
        combine_fold_output(test_rmsd_rmse, train_rmsd_rmse, test_rmsd_true, test_rmsd_pred, train_rmsd_true, train_rmsd_pred,
                            args.outprefix, args.test_interval, rmsd=True, second_data_source=False)

    if args.two_data_sources:
        if args.scoring:
            combine_fold_output(test2_auc, train2_auc, test2_y_true, test2_y_score, train2_y_true, train2_y_score,
                                args.outprefix, args.test_interval, affinity=False, second_data_source=True)

        if args.affinity:
            combine_fold_output(test2_aff_rmse, train2_aff_rmse, test2_aff_true, test2_aff_pred, train2_aff_true, train2_aff_pred,
                                args.outprefix, args.test_interval, affinity=True, second_data_source=True,
                                filter_actives_test=test2_y_true, filter_actives_train=train2_y_true)

        if args.rmsd:
            combine_fold_output(test2_rmsd_rmse, train2_rmsd_rmse, test2_rmsd_true, test2_rmsd_pred, train2_rmsd_true, train2_rmsd_pred,
                                args.outprefix, args.test_interval, rmsd=True, second_data_source=True)

