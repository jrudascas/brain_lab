from projects.qm_brain.utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

num_subjects = 15
main_path = '/home/user/Desktop/QMBrain/New Data/'
condition_list = ['Cond10/','Cond12/']


tpm_cond,hist_cond = [],[]

for condition in condition_list:

    hist_list,tpm_list = [],[]

    for i in range(num_subjects):

        subject_path = main_path + condition + str(i + 1) + '/'

        filenameTPM = subject_path + '/' + 'tpm_' + str(i + 1) + '.csv'
        filenameHist = subject_path + '/' + 'hist_' + str(i + 1) + '.csv'

        tpm_list.append(load_matrix(filenameTPM))
        hist_list.append(load_matrix(filenameHist))

    tpm_cond.append(np.squeeze(np.asarray(tpm_list)))
    hist_cond.append(np.squeeze(np.asarray(hist_list)))

tpm = np.array(tpm_cond)
hist = np.array(hist_cond)

cond10_avg_tpm = np.mean(tpm, axis=1)[0,...]
cond12_avg_tpm = np.mean(tpm, axis=1)[1,...]

cond10_avg_hist = np.mean(hist,axis=1)[0,...]
cond12_avg_hist = np.mean(hist,axis=1)[1,...]

unique10,count10 = np.unique(hist[0,...],return_counts=True)
unique12,count12 = np.unique(hist[1,...],return_counts=True)

count10 = count10/np.sum(count10)
count12 = count12/np.sum(count12)

f1 = plt.figure()
plt.bar(np.arange(len(count10)),count10)
plt.title('Histogram of 8 Regions Condition10')
plt.xlabel('Regions')
plt.xticks(np.arange(8),
           ['Occiptal R', 'Occiptal L', 'Parietal R', 'Parietal L', 'Posterior R', 'Posterior L', 'Anterior R',
            'Anterior L'])
plt.show()
plt.savefig('Cond10Hist')
plt.close()

f2 = plt.figure()
plt.bar(np.arange(len(count12)),count12)
plt.title('Histogram of 8 Regions Condition12')
plt.xlabel('Regions')
plt.xticks(np.arange(8),
           ['Occiptal R', 'Occiptal L', 'Parietal R', 'Parietal L', 'Posterior R', 'Posterior L', 'Anterior R',
            'Anterior L'])
plt.show()
plt.savefig('Cond12Hist')
plt.close()

f3 = plt.figure()
plt.imshow(cond10_avg_tpm, cmap='plasma', vmin=0, vmax=1)
plt.title('Transition Probability Matrix Condition10')
plt.colorbar()
plt.show()

f4 = plt.figure()
plt.imshow(cond12_avg_tpm, cmap='plasma', vmin=0, vmax=1)
plt.title('Transition Probability Matrix Condition12')
plt.colorbar()
plt.show()

statInd, pvalInd = ss.ttest_ind(count10,count12)
statRel,pvalRel = ss.ttest_rel(count10,count12)
statMW,pvalMW = ss.mannwhitneyu(count10,count12)

statIndTPM, pvalIndTPM = ss.ttest_ind(cond10_avg_tpm.flatten(),cond12_avg_tpm.flatten())
statRelTPM,pvalRelTPM = ss.ttest_rel(cond10_avg_tpm.flatten(),cond12_avg_tpm.flatten())
statMWTPM,pvalMWTPM = ss.mannwhitneyu(cond10_avg_tpm.flatten(),cond12_avg_tpm.flatten())



print('The t-test (independent) result is: ', statInd, 'with a p-value of: ', pvalInd)

print('The t-test (related) result is: ', statRel, 'with a p-value of: ', pvalRel)

print('The Mann-Whitney result is: ', statMW, 'with a p-value of: ', pvalMW)


print('The t-test TPM (independent) result is: ', statIndTPM, 'with a p-value of: ', pvalIndTPM)

print('The t-test TPM(related) result is: ', statRelTPM, 'with a p-value of: ', pvalRelTPM)

print('The Mann-Whitney TPM result is: ', statMWTPM, 'with a p-value of: ', pvalMWTPM)

