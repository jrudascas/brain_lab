from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

main_path = '/home/user/Desktop/QMBrain/'
condition_list = ['Cond10/','Cond12/']
exp_list = ['New Data/','BYD/','RestData/']

dim_list = [8]#[4,8]

save_path = '/home/user/Desktop/QMBrain/tpm/8_regions/'


for dim in dim_list:

    tpm_exp, hist_exp = [],[]

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

        tpm_cond, hist_cond = [], []

        for j in range(len(condition_list)):

            hist_list,tpm_list = [],[]

            for i in range(num_subjects):

                subject_path = exp_path + condition_list[j] + str(i + 1) + '/results'

                filenameTPM = subject_path + '/' + str(dim) + 'by' + str(dim) + 'tpm_' + str(i + 1) + '.csv'
                filenameHist = subject_path + '/' + str(dim) + 'by' + str(dim) + 'hist_' + str(i + 1) + '.csv'

                tpm_list.append(load_matrix(filenameTPM))
                hist_list.append(load_matrix(filenameHist))

            tpm_cond.append(tpm_list)
            hist_cond.append(hist_list)

        tpm_exp.append(np.squeeze(np.asarray(tpm_cond)))
        hist_exp.append(np.squeeze(np.asarray(hist_cond)))

    tpm = np.array(tpm_exp)
    hist = np.array(hist_exp)

    tpm_taken = tpm[0]
    tpm_byd = tpm[1]
    tpm_rest = tpm[2]

    hist_taken = hist[0]
    hist_byd = hist[1]
    hist_rest = hist[2]

    cond10_taken_avg_tpm = np.mean(tpm_taken, axis=1)[0, ...]
    cond12_taken_avg_tpm = np.mean(tpm_taken, axis=1)[1, ...]
    cond10_byd_avg_tpm = np.mean(tpm_byd, axis=1)[0, ...]
    cond12_byd_avg_tpm = np.mean(tpm_byd, axis=1)[1, ...]
    avg_tpm_rest = np.mean(tpm_rest, axis=0)

    cond10_taken_avg_hist = np.mean(hist_taken, axis=1)[0, ...]
    cond12_taken_avg_hist = np.mean(hist_taken, axis=1)[1, ...]
    cond10_byd_avg_hist = np.mean(hist_byd, axis=1)[0, ...]
    cond12_byd_avg_hist = np.mean(hist_byd, axis=1)[1, ...]
    avg_hist_rest = np.mean(hist_rest, axis=0)

    unique10_taken, count10_taken = np.unique(hist_taken[0, ...], return_counts=True)
    unique12_taken, count12_taken = np.unique(hist_taken[1, ...], return_counts=True)
    unique10_byd, count10_byd = np.unique(hist_byd[0, ...], return_counts=True)
    unique12_byd, count12_byd = np.unique(hist_byd[1, ...], return_counts=True)
    unique_rest, count_rest = np.unique(hist_rest, return_counts=True)

    count10_taken = count10_taken / np.sum(count10_taken)
    count12_taken = count12_taken / np.sum(count12_taken)
    count10_byd = count10_byd / np.sum(count10_byd)
    count12_byd = count12_byd / np.sum(count12_byd)
    count_rest = count_rest / np.sum(count_rest)

    f1 = plt.figure()
    plt.bar(np.arange(len(count10_taken)), count10_taken)
    plt.title('Histogram of 8 Regions Unscrambled Taken')
    plt.xlabel('Regions')
    plt.xticks(np.arange(8),
               ['Occiptal R', 'Occiptal L', 'Parietal R', 'Parietal L', 'Posterior R', 'Posterior L', 'Anterior R',
                'Anterior L'])
    plt.show()
    plt.savefig('Cond10Hist_taken')
    plt.close()

    f2 = plt.figure()
    plt.bar(np.arange(len(count12_taken)), count12_taken)
    plt.title('Histogram of 8 Regions Scrambled Taken')
    plt.xlabel('Regions')
    plt.xticks(np.arange(8),
               ['Occiptal R', 'Occiptal L', 'Parietal R', 'Parietal L', 'Posterior R', 'Posterior L', 'Anterior R',
                'Anterior L'])
    plt.show()
    plt.savefig('Cond12Hist_taken')
    plt.close()

    f3 = plt.figure()
    plt.bar(np.arange(len(count10_byd)), count10_byd)
    plt.title('Histogram of 8 Regions Unscrambled BYD')
    plt.xlabel('Regions')
    plt.xticks(np.arange(8),
               ['Occiptal R', 'Occiptal L', 'Parietal R', 'Parietal L', 'Posterior R', 'Posterior L', 'Anterior R',
                'Anterior L'])
    plt.show()
    plt.savefig('Cond10Hist_byd')
    plt.close()

    f4 = plt.figure()
    plt.bar(np.arange(len(count12_byd)), count12_byd)
    plt.title('Histogram of 8 Regions Scrambled BYD')
    plt.xlabel('Regions')
    plt.xticks(np.arange(8),
               ['Occiptal R', 'Occiptal L', 'Parietal R', 'Parietal L', 'Posterior R', 'Posterior L', 'Anterior R',
                'Anterior L'])
    plt.show()
    plt.savefig('Cond12Hist_byd')
    plt.close()

    f5 = plt.figure()
    plt.bar(np.arange(len(count_rest)), count_rest)
    plt.title('Histogram of 8 Regions Resting State')
    plt.xlabel('Regions')
    plt.xticks(np.arange(8),
               ['Occiptal R', 'Occiptal L', 'Parietal R', 'Parietal L', 'Posterior R', 'Posterior L', 'Anterior R',
                'Anterior L'])
    plt.show()
    plt.savefig('Hist_rest')
    plt.close()


    f6 = plt.figure()
    plt.imshow(cond10_taken_avg_tpm, cmap='plasma', vmin=0, vmax=1)
    plt.title('TPM Uncrambled Taken')
    plt.colorbar()
    plt.show()
    plt.savefig('tpm_taken_us')
    plt.close()

    f7 = plt.figure()
    plt.imshow(cond12_taken_avg_tpm, cmap='plasma', vmin=0, vmax=1)
    plt.title('TPM Scrambled Taken')
    plt.colorbar()
    plt.show()
    plt.savefig('tpm_taken_s')
    plt.close()

    f8 = plt.figure()
    plt.imshow(cond10_byd_avg_tpm, cmap='plasma', vmin=0, vmax=1)
    plt.title('TPM Uncrambled BYD')
    plt.colorbar()
    plt.show()
    plt.savefig('tpm_byd_us')
    plt.close()

    f9 = plt.figure()
    plt.imshow(cond12_byd_avg_tpm, cmap='plasma', vmin=0, vmax=1)
    plt.title('TPM Scrambled BYD')
    plt.colorbar()
    plt.show()
    plt.savefig('tpm_byd_s')
    plt.close()

    f10 = plt.figure()
    plt.imshow(avg_tpm_rest, cmap='plasma', vmin=0, vmax=1)
    plt.title('TPM Rest')
    plt.colorbar()
    plt.show()
    plt.savefig('tpm_rest')
    plt.close()

    save_file(count10_taken,save_path,'hist_us_taken')
    save_file(count12_taken, save_path, 'hist_s_taken')
    save_file(count10_byd, save_path, 'hist_us_byd')
    save_file(count12_byd, save_path, 'hist_s_byd')
    save_file(count_rest, save_path, 'hist_rest')

    save_file(cond10_taken_avg_tpm,save_path,'tpm_us_taken')
    save_file(cond12_taken_avg_tpm, save_path, 'tpm_s_taken')
    save_file(cond10_byd_avg_tpm, save_path, 'tpm_us_byd')
    save_file(cond12_byd_avg_tpm, save_path, 'tpm_s_byd')
    save_file(avg_tpm_rest, save_path, 'tpm_rest')





