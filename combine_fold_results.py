#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, re, argparse, sys, os
import sklearn.metrics
import scipy.stats
import seaborn as sns
import seaborn.utils, seaborn.palettes
from seaborn.palettes import blend_palette
from inspect import signature

def makejoint(x,y,color,title):
    '''Plot x vs y where x are experimental values and y are predictions'''
    color_rgb = matplotlib.colors.colorConverter.to_rgb(color)
    colors = [seaborn.utils.set_hls_values(color_rgb, l=l)
              for l in np.linspace(1, 0, 12)]
    cmap = blend_palette(colors, as_cmap=True)
    g = sns.JointGrid(x=x,y=y)
    #g = g.plot_joint(plt.hexbin, cmap=cmap)
    g = g.plot_joint(plt.scatter, marker='.',color=color,alpha=.25,s=2)
    g = g.plot_joint(sns.kdeplot, shade=True,cmap=cmap,shade_lowest=False)
    g = g.plot_marginals(sns.distplot, color=color)
    g.ax_joint.set_xlabel('Experiment')
    g.ax_joint.set_ylabel('Prediction')
    plt.suptitle(title,verticalalignment='center')
    s = scipy.stats.spearmanr(x,y)[0]
    rmse = np.sqrt(np.mean(np.square(np.array(x)-np.array(y))))
    ax = g.ax_joint
    ax.set_ylim(0,12)
    ax.set_xlim(0,12)
    #ax.plot(x,y,'.',color=color,alpha=.25,markersize=2)
    ax.text(0.5, 0.05, 'Spearman = %.3f, RMSE = %.3f'%(s,rmse),fontsize=18,transform=ax.transAxes,horizontalalignment='center')
    plt.savefig('%s.pdf'%title,bbox_inches='tight')

def read_results_file(file):
    '''Read columns of float data from a file, ignoring # comments'''
    rows = []
    with open(file, 'r') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if line:
                rows.append(list(map(float, line.split(' '))))
    return list(zip(*rows))


def write_results_file(file, *columns, **kwargs):
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

def plot_prc(plot_file, y_test, y_score, average_precision, txt):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_score)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('{0}  AP={1:0.2f}'.format(txt,
              average_precision))
    plt.axes().set_aspect('equal')
    plt.savefig(plot_file, bbox_inches='tight')


def plot_correlation(plot_file, y_aff, y_predaff, rmsd, r2):
    assert len(y_aff) == len(y_predaff)
    fig = plt.figure(figsize=(8,8))
    plt.plot(y_aff, y_predaff, 'o', alpha=.2,label='RMSD=%.2f, R^2=%.3f (Pos)' % (rmsd, r2))
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


def get_results_files(prefix, foldnums, two_data_sources):
    p, r, a = False, False, False
    files = {}
    if foldnums is None:
        foldnums = set()
        pattern = r'%s\.(\d+)\.(auc\.final(test|train)2?)$' % prefix
        for file in glob.glob(prefix + '*'):
            match = re.match(pattern, file)
            if match:
                foldnums.add(int(match.group(1)))
                p = True
        pattern = r'%s\.(\d+)\.(rmsd\.final(test|train)2?)$' % prefix
        for file in glob.glob(prefix + '*'):
            match = re.match(pattern, file)
            if match:
                foldnums.add(int(match.group(1)))
                a = True
        pattern = r'%s\.(\d+)\.(rmsd_rmse\.final(test|train)2?)$' % prefix
        for file in glob.glob(prefix + '*'):
            match = re.match(pattern, file)
            if match:
                foldnums.add(int(match.group(1)))
                r = True

    elif isinstance(foldnums, str):
        foldnums = [int(i) for i in foldnums.split(',') if i]
        pattern = r'%s\.(\d+)\.(auc\.final(test|train)2?)$' % prefix
        for file in glob.glob(prefix + '*'):
            match = re.match(pattern, file)
            if match:
                p = True
        pattern = r'%s\.(\d+)\.(rmsd\.final(test|train)2?)$' % prefix
        for file in glob.glob(prefix + '*'):
            match = re.match(pattern, file)
            if match:
                a = True
        pattern = r'%s\.(\d+)\.(rmsd_rmse\.final(test|train)2?)$' % prefix
        for file in glob.glob(prefix + '*'):
            match = re.match(pattern, file)
            if match:
                r = True
    
    for i in foldnums:
        files[i] = {}
        files[i]['out'] = '%s.%d.out' % (prefix, i)
        if p:
            files[i]['auc_finaltest'] = '%s.%d.auc.finaltest' % (prefix, i)
            files[i]['auc_finaltrain'] = '%s.%d.auc.finaltrain' % (prefix, i)
        if r:
            files[i]['rmsd_rmse_finaltest'] = '%s.%d.rmsd_rmse.finaltest' % (prefix, i)
            files[i]['rmsd_rmse_finaltrain'] = '%s.%d.rmsd_rmse.finaltrain' % (prefix, i)
        if a:
            files[i]['rmsd_finaltest'] = '%s.%d.rmsd.finaltest' % (prefix, i)
            files[i]['rmsd_finaltrain'] = '%s.%d.rmsd.finaltrain' % (prefix, i)
        if two_data_sources:
            if p:
                files[i]['auc_finaltest2'] = '%s.%d.auc.finaltest2' % (prefix, i)
                files[i]['auc_finaltrain2'] = '%s.%d.auc.finaltrain2' % (prefix, i)
            if a:
                files[i]['rmsd_finaltest2'] = '%s.%d.rmsd.finaltest2' % (prefix, i)
                files[i]['rmsd_finaltrain2'] = '%s.%d.rmsd.finaltrain2' % (prefix, i)
            if r:
                files[i]['rmsd_rmse_finaltest2'] = '%s.%d.rmsd_rmse.finaltest2' % (prefix, i)
                files[i]['rmsd_rmse_finaltrain2'] = '%s.%d.rmsd_rmse.finaltrain2' % (prefix, i)
    for i in files:
        for file in list(files[i].values()):
            check_file_exists(file)
    return files

def make_uniform_array(lists):
    '''Take a list of possibly variable sized lists and return a numpy 2D array
        where the smaller lists are padded to the size of the longest list using 
        their last value'''
    maxlen = max([len(l) for l in lists])
    return np.array([list(l) + [l[-1]]*(maxlen-len(l)) for l in lists])

def combine_fold_results(test_metrics, train_metrics, test_labels, test_preds, train_labels, train_preds,
                         outprefix, test_interval, file_type, second_data_source=False,
                         filter_actives_test=None, filter_actives_train=None):
    '''Make results files and graphs combined from results for
    separate crossvalidation folds. test_metrics and train_metrics
    are lists of lists of AUCs or RMSDs for each fold, for each
    test_interval. test_labels and test_preds are labels and final
    test predictions for each test fold, in a single list. train_labels
    and train_preds are labels and predictions for each train fold in
    a single list. affinity says whether to interpret the labels and
    predictions as affinities, and the metric as RMSD instead of AUC.
    second_data_source sets the output file names to reflect whether
    the results are for the second data source of a combined data model.'''
    if file_type=='affinity':
      metric = 'rmsd'
    elif file_type=='pose':
      metric='auc'
    elif file_type=='rmsd':
      metric = 'rmsd_rmse'
    two = '2' if second_data_source else ''

    #test and train metrics are organized by fold iteration, which with dynamic
    #stepping can be different lengths; pad them out to the max length with 
    #the last value
    test_metrics = make_uniform_array(test_metrics)
    train_metrics = make_uniform_array(train_metrics)
    
    #average metric across folds
    mean_test_metrics = np.mean(test_metrics, axis=0)
    mean_train_metrics = np.mean(train_metrics, axis=0)

    #write test and train metrics (mean and for each fold)
    write_results_file('%s.%s.test%s' % (outprefix, metric, two), mean_test_metrics, *test_metrics)
    write_results_file('%s.%s.train%s' % (outprefix, metric, two), mean_train_metrics, *train_metrics)

    #training plot of mean test and train metric across folds
    training_plot('%s_%s_train%s.pdf' % (outprefix, metric, two), mean_train_metrics, mean_test_metrics)
#added line
    
    if metric=='rmsd_rmse':

        if filter_actives_test:
            test_preds = np.array(test_preds)[np.array(test_labels)>0]
            test_labels = np.array(test_labels)[np.array(test_labels)>0]

        #correlation plots for last test iteration
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(test_preds, test_labels))
        r2 = scipy.stats.pearsonr(test_preds, test_labels)[0]**2
        write_results_file('%s.rmsd_rmse.finaltest%s' % (outprefix, two), test_labels, test_preds, footer='RMSE,R %f %f\n' % (rmse, r2))
        makejoint(test_labels, test_preds, 'royalblue','%s_corr_test%s' % (outprefix, two))
        #plot_correlation('%s_corr_test%s.pdf' % (outprefix, two), test_preds, test_labels, rmse, r2)

        if filter_actives_train:
            train_preds = np.array(train_preds)[np.array(train_labels)>0]
            train_labels = np.array(train_labels)[np.array(train_labels)>0]

        rmse = np.sqrt(sklearn.metrics.mean_squared_error(train_preds, train_labels))
        r2 = scipy.stats.pearsonr(train_preds, train_labels)[0]**2
        write_results_file('%s.rmsd_rmse.finaltrain%s' % (outprefix, two), train_labels, train_preds, footer='RMSE,R %f %f\n' % (rmse, r2))
        makejoint(train_labels, train_preds, 'orangered','%s_corr_train%s' % (outprefix, two))        
        #plot_correlation('%s_corr_train%s.pdf' % (outprefix, two), train_preds, train_labels, rmse, r2)
    
    elif metric=='rmsd':

        if filter_actives_test:
            test_preds = np.array(test_preds)[np.array(test_labels)>0]
            test_labels = np.array(test_labels)[np.array(test_labels)>0]

        #correlation plots for last test iteration
        rmsd = np.sqrt(sklearn.metrics.mean_squared_error(test_preds, test_labels))
        r2 = scipy.stats.pearsonr(test_preds, test_labels)[0]**2
        write_results_file('%s.rmsd.finaltest%s' % (outprefix, two), test_labels, test_preds, footer='RMSD,R %f %f\n' % (rmsd, r2))
        makejoint(test_labels, test_preds, 'royalblue','%s_corr_test%s' % (outprefix, two))
        #plot_correlation('%s_corr_test%s.pdf' % (outprefix, two), test_preds, test_labels, rmsd, r2)

        if filter_actives_train:
            train_preds = np.array(train_preds)[np.array(train_labels)>0]
            train_labels = np.array(train_labels)[np.array(train_labels)>0]

        rmsd = np.sqrt(sklearn.metrics.mean_squared_error(train_preds, train_labels))
        r2 = scipy.stats.pearsonr(train_preds, train_labels)[0]**2
        write_results_file('%s.rmsd.finaltrain%s' % (outprefix, two), train_labels, train_preds, footer='RMSD,R %f %f\n' % (rmsd, r2))
        makejoint(train_labels, train_preds, 'orangered','%s_corr_train%s' % (outprefix, two))        
        #plot_correlation('%s_corr_train%s.pdf' % (outprefix, two), train_preds, train_labels, rmsd, r2)
    
    elif metric=='auc':

        #roc curves for the last test iteration
        if len(np.unique(test_labels)) > 1:
            last_iters = 1000
            avg_auc, max_auc, min_auc = last_iters_statistics(test_metrics, test_interval, last_iters)
            txt = 'For the last %s iterations:\nmean AUC=%.2f  max AUC=%.2f  min AUC=%.2f' % (last_iters, avg_auc, max_auc, min_auc)
            fpr, tpr, _ = sklearn.metrics.roc_curve(test_labels, test_preds)
            auc = sklearn.metrics.roc_auc_score(test_labels, test_preds)
            precscore = sklearn.metrics.average_precision_score(test_labels, test_preds)
            write_results_file('%s.auc.finaltest%s' % (outprefix, two), test_labels, test_preds, footer='AUC, precision score %f %f\n' % (auc,precscore))
            plot_roc_curve('%s_roc_test%s.pdf' % (outprefix, two), fpr, tpr, auc, txt)
            plot_prc('%s_prc_test%s.pdf' % (outprefix, two),test_labels, test_preds, precscore, '%s.auc.finaltest%s' % (outprefix, two))

        if len(np.unique(train_labels)) > 1:
            last_iters = 1000
            avg_auc, max_auc, min_auc = last_iters_statistics(train_metrics, test_interval, last_iters)
            txt = 'For the last %s iterations:\nmean AUC=%.2f  max AUC=%.2f  min AUC=%.2f' % (last_iters, avg_auc, max_auc, min_auc)
            fpr, tpr, _ = sklearn.metrics.roc_curve(train_labels, train_preds)
            auc = sklearn.metrics.roc_auc_score(train_labels, train_preds)
            precscore = sklearn.metrics.average_precision_score(train_labels, train_preds)
            write_results_file('%s.auc.finaltrain%s' % (outprefix, two), train_labels, train_preds, footer='AUC, precision score %f %f\n' % (auc,precscore))
            plot_roc_curve('%s_roc_train%s.pdf' % (outprefix, two), fpr, tpr, auc, txt)
            plot_prc('%s_prc_train%s.pdf' % (outprefix, two),train_labels, train_preds, precscore, '%s.auc.finaltrain%s' % (outprefix, two))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Combine training results from different folds and make graphs')
    parser.add_argument('-o','--outprefix',type=str,required=True,help="Prefix for input and output files (--outprefix from train.py)")
    parser.add_argument('-n','--foldnums',type=str,required=False,help="Fold numbers to combine, default is to determine using glob",default=None)
    parser.add_argument('-2','--two_data_sources',default=False,action='store_true',required=False,help="Whether to look for 2nd data source results files")
    parser.add_argument('-t','--test_interval',type=int,default=40,required=False,help="Number of iterations between tests")
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()

    try:
        results_files = get_results_files(args.outprefix, args.foldnums, args.two_data_sources)
    except OSError as e:
        print("error: %s" % e)
        sys.exit(1)

    if len(results_files) == 0:
        print("error: missing results files")
        sys.exit(1)

    for i in results_files:
        for key in sorted(results_files[i], key=len):
            print(str(i).rjust(3), key.rjust(15), results_files[i][key])

    test_aucs, train_aucs = [], []
    test_rmsd_rmses, train_rmsd_rmses = [], []
    test_rmsds, train_rmsds = [], []
    test_y_true, train_y_true = [], []
    test_y_score, train_y_score = [], []
    test_y_aff, train_y_aff = [], []
    test_y_predaff, train_y_predaff = [], []
    test_y_rmsd, train_y_rmsd = [], []
    test_y_predrmsd, train_y_predrmsd = [], []
    test2_aucs, train2_aucs = [], []
    test2_rmsd_rmses, train2_rmsd_rmses = [], []
    test2_y_true, train2_y_true = [], []
    test2_y_score, train2_y_score = [], []
    test2_y_aff, train2_y_aff = [], []
    test2_y_predaff, train2_y_predaff = [], []
    test2_y_rmsd, train2_y_rmsd = [], []
    test2_y_predrmsd, train2_y_predrmsd = [], []
    pose, affinity, rmsd = False, False, False  
    
    #read results files
    for i in results_files:

        if (('auc_finaltest' in results_files[i]) or ('auc_finaltrain' in results_files[i])):
            pose = True
            y_true, y_score = read_results_file(results_files[i]['auc_finaltest'])
            test_y_true.extend(y_true)
            test_y_score.extend(y_score)
            y_true, y_score = read_results_file(results_files[i]['auc_finaltrain'])
            train_y_true.extend(y_true)
            train_y_score.extend(y_score)

        if (('rmsd_rmse_finaltest' in results_files[i]) or ('rmsd_rmse_finaltrain' in results_files[i])):
            rmsd = True
            y_rmsd, y_predrmsd = read_results_file(results_files[i]['rmsd_rmse_finaltest'])
            test_y_rmsd.extend(y_rmsd)
            test_y_predrmsd.extend(y_predrmsd)
            y_rmsd, y_predrmsd = read_results_file(results_files[i]['rmsd_rmse_finaltrain'])
            train_y_rmsd.extend(y_rmsd)
            train_y_predrmsd.extend(y_predrmsd)

        if (('rmsd_finaltest' in results_files[i]) or ('rmsd_finaltrain' in results_files[i])):
            affinity = True
            y_aff, y_predaff = read_results_file(results_files[i]['rmsd_finaltest'])
            test_y_aff.extend(y_aff)
            test_y_predaff.extend(y_predaff)
            y_aff, y_predaff = read_results_file(results_files[i]['rmsd_finaltrain'])
            train_y_aff.extend(y_aff)
            train_y_predaff.extend(y_predaff)

        if args.two_data_sources:
            if (('auc_finaltest2' in results_files) or ('auc_finaltrain2' in results_files)):
                pose = True
                y_true, y_score = read_results_file(results_files[i]['auc_finaltest2'])
                test2_y_true.extend(y_true)
                test2_y_score.extend(y_score)
                y_true, y_score = read_results_file(results_files[i]['auc_finaltrain2'])
                train2_y_true.extend(y_true)
                train2_y_score.extend(y_score)

            if (('rmsd_finaltest2' in results_files) or ('rmsd_finaltrain2' in results_files)):
                affinity = True
                y_aff, y_predaff = read_results_file(results_files[i]['rmsd_finaltest2'])
                test2_y_aff.extend(y_aff)
                test2_y_predaff.extend(y_predaff)
                y_aff, y_predaff = read_results_file(results_files[i]['rmsd_finaltrain2'])
                train2_y_aff.extend(y_aff)
                train2_y_predaff.extend(y_predaff)

            if (('rmsd_rmse_finaltest2' in results_files[i]) or ('rmsd_rmse_finaltrain2' in results_files[i])):
                rmsd = True
                y_rmsd, y_predrmsd = read_results_file(results_files[i]['rmsd_rmse_finaltest2'])
                test2_y_rmsd.extend(y_rmsd)
                test2_y_predrmsd.extend(y_predrmsd)
                y_rmsd, y_predrmsd = read_results_file(results_files[i]['rmsd_rmse_finaltrain2'])
                train2_y_rmsd.extend(y_rmsd)
                train2_y_predrmsd.extend(y_predrmsd)

        #[test_auc train_auc train_loss] lr [test_rmsd train_rmsd] [[test2_auc train2_auc train2_loss] [test2_rmsd train2_rmsd]]
        out_columns = read_results_file(results_files[i]['out'])

        col_idx = 0
        if pose:
            test_aucs.append(out_columns[col_idx])
            train_aucs.append(out_columns[col_idx+1])
            #ignore train_loss
            col_idx += 3

        #ignore learning rate
        col_idx += 1

        if affinity:
            test_rmsds.append(out_columns[col_idx])
            train_rmsds.append(out_columns[col_idx+1])
            col_idx += 2
        if rmsd:
            test_rmsd_rmses.append(out_columns[col_idx])
            train_rmsd_rmses.append(out_columns[col_idx+1])
            col_idx += 2

        if args.two_data_sources:
            if pose:
                test2_aucs.append(out_columns[col_idx])
                train2_aucs.append(out_columns[col_idx+1])
                #ignore train2_loss
                col_idx += 3

            if affinity:
                test2_rmsds.append(out_columns[col_idx])
                train2_rmsds.append(out_columns[col_idx+1])
                col_idx += 2

            if rmsd:
                test2_rmsd_rmses.append(out_columns[col_idx])
                train2_rmsd_rmses.append(out_columns[col_idx+1])
                col_idx += 2 

    if pose:
        combine_fold_results(test_aucs, train_aucs, test_y_true, test_y_score, train_y_true, train_y_score,
                             args.outprefix, args.test_interval, 'pose', second_data_source=False)

    if affinity:
        combine_fold_results(test_rmsds, train_rmsds, test_y_aff, test_y_predaff, train_y_aff, train_y_predaff,
                             args.outprefix, args.test_interval, 'affinity', second_data_source=False,
                             filter_actives_test=test_y_true, filter_actives_train=train_y_true)
    if rmsd:
        combine_fold_results(test_rmsd_rmses, train_rmsd_rmses, test_y_rmsd, test_y_predrmsd, train_y_rmsd, train_y_predrmsd,
                             args.outprefix, args.test_interval, 'rmsd', second_data_source=False)


    if args.two_data_sources:
        if pose:
            combine_fold_results(test2_aucs, train2_aucs, test2_y_true, test2_y_score, train2_y_true, train2_y_score,
                                 args.outprefix, args.test_interval, 'pose', second_data_source=True)
        if affinity:
            combine_fold_results(test2_rmsds, train2_rmsds, test2_y_aff, test2_y_predaff, train2_y_aff, train2_y_predaff,
                                 args.outprefix, args.test_interval, 'affinity', second_data_source=True,
                                 filter_actives_test=test2_y_true, filter_actives_train=train2_y_true)
        if rmsd:
            combine_fold_results(test2_rmsd_rmses, train2_rmsd_rmses, test2_y_rmsd, test2_y_predrmsd, train2_y_rmsd, train2_y_predrmsd,
                             args.outprefix, args.test_interval, 'rmsd', second_data_source=True)

