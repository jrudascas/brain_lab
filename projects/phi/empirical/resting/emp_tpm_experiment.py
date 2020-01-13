import numpy as np
import matplotlib.pyplot as plt
from projects.phi.tools.utils import *
from scipy.stats import mannwhitneyu
from scipy.stats import entropy
import numpy as np
import networkx as nx
from projects.generalize_ising_model.core import generalized_ising
from projects.generalize_ising_model.tools.utils import to_normalize, to_save_results, correlation_function, dim, find_nearest

brain_states = ['Awake','Deep','Mild','Recovery']

networks = ['Aud','DMN','Dorsal','Ventral','Cingulo','Fronto','Retro','SMhand','SMmouth','Vis']

main_path = '/home/user/Desktop/data_phi/phi/'

totalMatrix,stateMatrix,addMatrix = [],[],[]

for network in networks:
    for state in brain_states:
        tpm = load_matrix(main_path + state + '/SbyS/' + network + '/' + state +'tpm.npy')
        for m in range(tpm.shape[0]):
            addMatrix.append(tpm[m,...])
        stateMatrix.append(addMatrix)
        addMatrix=[]

    totalMatrix.append(stateMatrix)
    stateMatrix = []

totalMatrix = np.asarray(totalMatrix)


#plt.violinplot(totalMatrix[1,:,:,31,2].T,[0,1,2,3])
#plt.show()

number_bins = 100
temperature_parameters = (0.05, 10, 50)
ts = np.linspace(temperature_parameters[0],
                 temperature_parameters[1],
                 temperature_parameters[2])

main_save_path = '/home/user/Desktop/data_phi/tpm/'

for i in range(totalMatrix.shape[0]):

    print(networks[i])
    S_conditions = []
    den_conditions = []
    dimensionality_conditions = []
    hubs_conditions = []

    save_path = main_save_path + networks[i]

    for condition in range(totalMatrix.shape[1]):
        S_list = []
        D_list = []
        dim_list = []
        hubs_list = []
        for tpm_index in range(totalMatrix.shape[2]):

            tpm_ = totalMatrix[i,condition,tpm_index, ...]
            hist, bin_edges = np.histogram(tpm_index, number_bins)

            hist = hist/np.sum(tpm_)

            S_list.append(entropy(hist))
            D_list.append(nx.density(nx.Graph(tpm_)))


            simulated_fc, critical_temperature, E, M, S, H = generalized_ising(tpm_, temperature_parameters=temperature_parameters, n_time_points=1200, thermalize_time=0.3, temperature_distribution ='lineal', phi_variables = False, type='digital')

            c, r = correlation_function(simulated_fc, tpm_)
            dimensionality = dim(c, r, find_nearest(ts, critical_temperature))

            dim_list.append(dimensionality)

            h = nx.hits(nx.Graph(tpm_))[0]

            hubs_list.append([value for key, value in h.items()])

        S_conditions.append(S_list)
        den_conditions.append(D_list)
        dimensionality_conditions.append(dim_list)
        hubs_conditions.append(hubs_list)

    viola_plot_n_save(S_conditions,save_path,'Entropy')
    viola_plot_n_save(den_conditions,save_path,'Density')
    viola_plot_n_save(dimensionality_conditions,save_path, 'Dimensionality')
    save_hubs(hubs_conditions,save_path)
    '''
    plt.violinplot(S_conditions, np.arange(len(S_conditions)))
    plt.title('Entropy of TPM')
    plt.show()
    plt.violinplot(den_conditions, np.arange(len(den_conditions)))
    plt.title('Density of TPM')
    plt.show()
    plt.violinplot(dimensionality_conditions, np.arange(len(dimensionality_conditions)))
    plt.title('Dimensionality of TPM')
    plt.show()
    

    save_list(S_conditions,'/home/user/Desktop/data_phi/tpm/',brain_states)
    save_list(den_conditions)
    save_list(dimensionality_conditions)
    save_list(hubs_conditions)
    '''






#0.05/(32*5)