from projects.qm_brain.utils.utils import *
import numpy as np

tpm_dims  = [8]#[4,8]

#experiment_type = ['RestData/']

experiment_type = ['New Data/','BYD/','RestData/']

main_path = '/home/user/Desktop/QMBrain/'

condition_list = ['Cond10/','Cond12/']

occipital_R_ind = np.array([61 ,56 ,57 ,62 ,66 ,63 ,67 ,70,73,55])
# occipital_C_ind = np.array([54 ,55])
# I arbitrarily put 55(72) on the right and 54(75) on the left
occipital_L_ind = np.array([52, 53, 51, 50, 49, 46, 45, 44,43,54])

parietal_R_ind = np.array([58 ,59, 60 ,65 ,64 ,68 ,69 ,77 ,72 ,71 ,74 ,75 ,76 ,81 ,80 ,85,42])
# parietal_C_ind = np.array([48, 42])
# I arbitrarily put 42(55) on the right and 48(62) on the left
parietal_L_ind = np.array([47 ,41 ,24 ,29 ,40 ,39 ,33 ,28 ,36 ,38 ,37 ,35 ,32 ,31 ,34 ,30,48])

posteriorf_R_ind = np.array([79 ,78 ,84 ,3 ,88 ,83 ,82 ,87 ,91 ,86,4])
#posteriorf_C_ind = np.array([4])
# I arbitrarily put 4(6) on the right
posteriorf_L_ind = np.array([5 ,23 ,10 ,9 ,15 ,22 ,27 ,21 ,18 ,26])

anteriorf_R_ind = np.array([2 ,7 ,6 ,1 ,90 ,0 ,89 , 11])
# anteriorf_C_ind = np.array([8 ,12 ,11])
# I arbitraily put 8(11) and 12(16) on the left, and 11(15) on the right
anteriorf_L_ind = np.array([14 ,13 ,16 ,17 ,20 ,19 ,25,8,12])

for dim in tpm_dims:
    for exp in experiment_type:
        if exp == 'New Data/':
            num_subjects = 15
        elif exp == 'BYD/':
            num_subjects = 14
        elif exp == 'RestData/':
            num_subjects=13
            condition_list = ['/']
        else:
            num_subjects = []

        exp_path = main_path + exp

        for condition in condition_list:

            for i in range(num_subjects):

                subject_path = exp_path + condition + str(i + 1) + '/'

                print('Running for subject ', i + 1, 'in folder ', exp_path + condition)

                filepathData = subject_path + 'data.csv'

                data = load_matrix(filepathData)

                phase, normAmp, probability = process_eeg_data(data)

                occipR = np.sum(probability[:,occipital_R_ind],axis=-1)
                occipL = np.sum(probability[:, occipital_L_ind], axis=-1)
                #occipC = np.sum(probability[:, occipital_C_ind], axis=-1)

                occip_tot = occipL+occipR

                parR = np.sum(probability[:, parietal_R_ind], axis=-1)
                parL = np.sum(probability[:, parietal_L_ind], axis=-1)
                #parC = np.sum(probability[:, parietal_C_ind], axis=-1)

                par_tot = parL + parR

                postR = np.sum(probability[:, posteriorf_R_ind], axis=-1)
                postL = np.sum(probability[:, posteriorf_L_ind], axis=-1)
                #postC = np.sum(probability[:, posteriorf_C_ind], axis=-1)

                post_tot = postL + postR

                antR = np.sum(probability[:, anteriorf_R_ind], axis=-1)
                antL = np.sum(probability[:, anteriorf_L_ind], axis=-1)
                #antC = np.sum(probability[:, anteriorf_C_ind], axis=-1)

                ant_tot = antL + antR

                sect_probs = np.array([occipR,occipL,parR,parL,postR,postL,antR,antL])
                large_sect_probs = np.array([occip_tot,par_tot,post_tot,ant_tot])

                if dim == 8:

                    max_inds = np.argmax(sect_probs,axis=0)

                    m = np.max(max_inds)+1

                    tpm = np.zeros((m,m))
                    norm = np.zeros((m,1))

                    max_inds_list = max_inds.tolist()

                    for (s1, s2) in zip(max_inds_list, max_inds_list[1:]):
                            norm[s1]+=1
                            tpm[s1][s2] += 1

                    tpm = tpm/norm
                    tpm[np.isnan(tpm)] = 0

                    labels = ['Occiptal R','Occiptal L','Parietal R','Parietal L','Posterior R','Posterior L','Anterior R','Anterior L']

                elif dim == 4:
                    max_inds = np.argmax(large_sect_probs, axis=0)

                    m = np.max(max_inds) + 1

                    tpm = np.zeros((m, m))
                    norm = np.zeros((m, 1))

                    max_inds_list = max_inds.tolist()

                    for (s1, s2) in zip(max_inds_list, max_inds_list[1:]):
                        norm[s1] += 1
                        tpm[s1][s2] += 1

                    tpm = tpm / norm
                    tpm[np.isnan(tpm)] = 0

                    labels = ['Occiptal', 'Parietal', 'Posterior', 'Anterior']

                else:
                    print('Invalid Dimensions for TPM')
                    max_inds = []
                    break

                if makedir2(subject_path):
                    default_delimiter = ','
                    format = '%1.5f'
                    filenameTPM = subject_path + '/results/' + str(dim) + 'by' + str(dim) + 'tpm_' + str(i+1) + '.csv'
                    filenameTPM_fig = subject_path + '/results/' + str(dim) + 'by' + str(dim) + 'tpm_' + str(i+1) + '.png'
                    filenameHist = subject_path + '/results/' + str(dim) + 'by' + str(dim) + 'hist_' + str(i + 1) + '.csv'
                    filenameHist_fig = subject_path + '/results/' + str(dim) + 'by' + str(dim) + 'hist_' + str(i + 1) + '.png'

                    if not file_exists(filenameTPM):
                        np.savetxt(filenameTPM, tpm, delimiter=default_delimiter, fmt=format)
                    if not file_exists(filenameHist):
                        np.savetxt(filenameHist, max_inds, delimiter=default_delimiter, fmt=format)

                    if not file_exists(filenameTPM_fig):
                        f1 = plt.figure()
                        plt.imshow(tpm, cmap='plasma', vmin=0, vmax=1)
                        plt.title('Transition Probability Matrix')
                        plt.xticks(np.arange(len(labels)), labels)
                        plt.colorbar()
                        plt.savefig(filenameTPM_fig,dpi=1200)
                        plt.close(f1)

                    if not file_exists(filenameHist_fig):
                        f2 = plt.figure()
                        plt.hist(max_inds, density=True, bins=8)
                        plt.title('Histogram of ' + str(dim) + ' Regions')
                        plt.xlabel('Regions')
                        plt.xticks(np.arange(len(labels)), labels)
                        plt.savefig(filenameHist_fig,dpi=1200)
                        plt.close(f2)




