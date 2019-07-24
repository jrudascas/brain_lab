from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt

num_subjects = 15
main_path = '/home/user/Desktop/QMBrain/New Data/'

condition_list = ['Cond10/', 'Cond12/']

occipital_R_ind = np.array([61, 56, 57, 62, 66, 63, 67, 70, 73])
occipital_C_ind = np.array([54, 55])
occipital_L_ind = np.array([52, 53, 51, 50, 49, 46, 45, 44, 43])

parietal_R_ind = np.array([58, 59, 60, 65, 64, 68, 69, 77, 72, 71, 74, 75, 76, 81, 80, 85])
parietal_C_ind = np.array([48, 42])
parietal_L_ind = np.array([47, 41, 24, 29, 40, 39, 33, 28, 36, 38, 37, 35, 32, 31, 34, 30])

posteriorf_R_ind = np.array([79, 78, 84, 3, 88, 83, 82, 87, 91, 86])
posteriorf_C_ind = np.array([4])
posteriorf_L_ind = np.array([5, 23, 10, 9, 15, 22, 27, 21, 18, 26])

anteriorf_R_ind = np.array([2, 7, 6, 1, 90, 0, 89])
anteriorf_C_ind = np.array([8, 12, 11])
anteriorf_L_ind = np.array([14, 13, 16, 17, 20, 19, 25])

for condition in condition_list:

    for i in range(num_subjects):

        subject_path = main_path + condition + str(i + 1) + '/'

        print('Running for subject ', i + 1, 'in folder ', condition)

        filepathData = subject_path + 'data.csv'

        data = load_matrix(filepathData)

        phase, normAmp, probability = process_eeg_data(data)

        occipR = np.sum(probability[:, occipital_R_ind], axis=-1)
        occipL = np.sum(probability[:, occipital_L_ind], axis=-1)
        occipC = np.sum(probability[:, occipital_C_ind], axis=-1)

        occip_tot = occipL + occipR

        parR = np.sum(probability[:, parietal_R_ind], axis=-1)
        parL = np.sum(probability[:, parietal_L_ind], axis=-1)
        parC = np.sum(probability[:, parietal_C_ind], axis=-1)

        par_tot = parL + parR

        postR = np.sum(probability[:, posteriorf_R_ind], axis=-1)
        postL = np.sum(probability[:, posteriorf_L_ind], axis=-1)
        postC = np.sum(probability[:, posteriorf_C_ind], axis=-1)

        post_tot = postL + postR

        antR = np.sum(probability[:, anteriorf_R_ind], axis=-1)
        antL = np.sum(probability[:, anteriorf_L_ind], axis=-1)
        antC = np.sum(probability[:, anteriorf_C_ind], axis=-1)

        ant_tot = antL + antR

        sect_probs = np.array([occipR, occipC,occipL, parR, parC, parL, postR, postC, postL, antR, antC, antL])
        large_sect_probs = np.array([occip_tot, par_tot, post_tot, ant_tot])

        max_inds = np.argmax(sect_probs, axis=0)

        m = np.max(max_inds) + 1

        tpm = np.zeros((m, m))
        norm = np.zeros((m, 1))

        max_inds_list = max_inds.tolist()

        for (s1, s2) in zip(max_inds_list, max_inds_list[1:]):
            norm[s1] += 1
            tpm[s1][s2] += 1

        tpm = tpm / norm
        tpm[np.isnan(tpm)] = 0

        if makedir2(subject_path):
            default_delimiter = ','
            format = '%1.5f'
            filenameTPM = subject_path + '/' + 'tpm_' + str(i + 1) + '.csv'
            filenameHist = subject_path + '/' + 'hist_' + str(i + 1) + '.csv'

            if not file_exists(filenameTPM):
                np.savetxt(filenameTPM, tpm, delimiter=default_delimiter, fmt=format)

            if not file_exists(filenameHist):
                np.savetxt(filenameHist, max_inds, delimiter=default_delimiter, fmt=format)

        f1 = plt.figure()
        plt.hist(max_inds, density=True, bins=13)
        plt.title('Histogram of 12 Regions')
        plt.xlabel('Regions')
        plt.xticks(np.arange(12),
                   ['Occiptal R', 'Occipital C','Occiptal L', 'Parietal R', 'Parietal C','Parietal L', 'Posterior R', 'Posterior C','Posterior L', 'Anterior R', 'Anterior C',
                    'Anterior L'])
        plt.show()

        f2 = plt.figure()
        plt.imshow(tpm, cmap='plasma', vmin=0, vmax=1)
        plt.title('Transition Probability Matrix')
        plt.xticks(np.arange(12),
                   ['Occiptal R', 'Occipital C','Occiptal L', 'Parietal R', 'Parietal C','Parietal L', 'Posterior R', 'Posterior C','Posterior L', 'Anterior R', 'Anterior C',
                    'Anterior L'])
        plt.colorbar()
        plt.show()



