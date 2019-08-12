from projects.qm_brain.utils.utils import *
import scipy.stats as ss
import numpy as np
import pandas as pd

cond10_taken_avg_tpm = load_matrix('/home/user/Desktop/QMBrain/tpm/8_regions/tpm_us_taken.csv')
cond12_taken_avg_tpm = load_matrix('/home/user/Desktop/QMBrain/tpm/8_regions/tpm_s_taken.csv')
cond10_byd_avg_tpm = load_matrix('/home/user/Desktop/QMBrain/tpm/8_regions/tpm_us_byd.csv')
cond12_byd_avg_tpm = load_matrix('/home/user/Desktop/QMBrain/tpm/8_regions/tpm_s_byd.csv')
avg_tpm_rest = load_matrix('/home/user/Desktop/QMBrain/tpm/8_regions/tpm_rest.csv')

cond10_taken_avg_hist = load_matrix('/home/user/Desktop/QMBrain/tpm/8_regions/hist_us_taken.csv')
cond12_taken_avg_hist = load_matrix('/home/user/Desktop/QMBrain/tpm/8_regions/hist_s_taken.csv')
cond10_byd_avg_hist = load_matrix('/home/user/Desktop/QMBrain/tpm/8_regions/hist_us_byd.csv')
cond12_byd_avg_hist = load_matrix('/home/user/Desktop/QMBrain/tpm/8_regions/hist_s_byd.csv')
avg_hist_rest = load_matrix('/home/user/Desktop/QMBrain/tpm/8_regions/hist_rest.csv')

tpm_list = [cond10_taken_avg_tpm,cond12_taken_avg_tpm,cond10_byd_avg_tpm,cond12_byd_avg_tpm,avg_tpm_rest]
hist_list = [cond10_taken_avg_hist,cond12_taken_avg_hist,cond10_byd_avg_hist,cond12_byd_avg_hist,avg_hist_rest]

types = ['Taken US','Taken S','BYD US','BYD S','Rest']

comparison_inds_1, comparison_inds_2 = np.triu_indices(5,k=1)

x = ss.ttest_ind(cond10_taken_avg_hist,avg_hist_rest)


for ind1,ind2 in zip(comparison_inds_1,comparison_inds_2):

    statIndTPM, pvalIndTPM = ss.ttest_ind(tpm_list[ind1].flatten(), tpm_list[ind2].flatten())
    statRelTPM, pvalRelTPM = ss.ttest_rel(tpm_list[ind1].flatten(), tpm_list[ind2].flatten())
    statMWTPM, pvalMWTPM = ss.mannwhitneyu(tpm_list[ind1].flatten(), tpm_list[ind2].flatten())

    print('The t-test TPM (independent) result is: ', statIndTPM, 'with a p-value of: ', pvalIndTPM,
          'for the comparison: ' + types[ind1] + ' vs. ' + types[ind2])

    print('The t-test TPM(related) result is: ', statRelTPM, 'with a p-value of: ', pvalRelTPM,
          'for the comparison: ' + types[ind1] + ' vs. ' + types[ind2])

    print('The Mann-Whitney TPM result is: ', statMWTPM, 'with a p-value of: ', pvalMWTPM,
          'for the comparison: ' + types[ind1] + ' vs. ' + types[ind2])

    print('\n ')

    statInd, pvalInd = ss.ttest_ind(hist_list[ind1], hist_list[ind2])
    statRel, pvalRel = ss.ttest_rel(hist_list[ind1], hist_list[ind2])
    statMW, pvalMW = ss.mannwhitneyu(hist_list[ind1], hist_list[ind2])

    print('The t-test (independent) result is: ', statInd, 'with a p-value of: ', pvalInd,
          'for the comparison: ' + types[ind1] + ' vs. ' + types[ind2])

    print('The t-test (related) result is: ', statRel, 'with a p-value of: ', pvalRel, 'for the comparison: ' + types[ind1] + ' vs. ' + types[ind2])

    print('The Mann-Whitney result is: ', statMW, 'with a p-value of: ', pvalMW, 'for the comparison: ' + types[ind1] + ' vs. ' + types[ind2])

    print('\n ')




