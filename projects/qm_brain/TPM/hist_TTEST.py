from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd

main_path = '/home/user/Desktop/QMBrain/'
condition_list = ['Cond10/','Cond12/']
exp_list = ['New Data/','BYD/','RestData/']

dim_list = [8]#[4,8]

save_path = '/home/user/Desktop/QMBrain/tpm/8_regions/'


for dim in dim_list:

    hist_exp = []

    for exp in exp_list:
        if exp == 'New Data/':
            num_subjects = 15
        elif exp == 'BYD/':
            num_subjects = 14
        elif exp == 'RestData/':
            num_subjects = 13
            condition_list = ['/']
        else:
            num_subjects=[]

        exp_path = main_path + exp

        hist_cond = []

        for j in range(len(condition_list)):

            hist_list = []

            for i in range(num_subjects):

                subject_path = exp_path + condition_list[j] + str(i + 1) + '/results'

                filenameHist = subject_path + '/' + str(dim) + 'by' + str(dim) + 'hist_' + str(i + 1) + '.csv'

                hist_list.append(load_matrix(filenameHist))

            hist_cond.append(hist_list)

        hist_exp.append(np.squeeze(np.asarray(hist_cond)))


    hist = np.array(hist_exp)

    hist_taken_us = hist[0][0,...]
    hist_taken_s = hist[0][1, ...]
    hist_byd_us = hist[1][0,...]
    hist_byd_s = hist[1][1, ...]
    hist_rest = hist[2]

    hists = [hist_taken_us,hist_taken_s,hist_byd_us,hist_byd_s,hist_rest]

    types = ['Taken US', 'Taken S', 'BYD US', 'BYD S', 'Rest']

    hist_by_type = []
    for h in hists:
        hist_by_sub = []
        for sub in range(h.shape[0]):
            unique_sub,count_sub = np.unique(h[sub,...],return_counts=True)
            hist_by_sub.append(count_sub/np.sum(count_sub))
        hist_by_type.append(np.array(hist_by_sub))

    total_hist = np.array(hist_by_type)

    comparison_inds_1, comparison_inds_2 = np.triu_indices(5, k=1)

    for ind1, ind2 in zip(comparison_inds_1, comparison_inds_2):
        statInd, pvalInd = ss.ttest_ind(total_hist[ind1], total_hist[ind2])

        print('The t-test (independent) result is: ', statInd, 'with a p-value of: ', pvalInd,
              'for the comparison: ' + types[ind1] + ' vs. ' + types[ind2])

        print('\n ')

