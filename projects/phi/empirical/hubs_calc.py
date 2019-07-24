import numpy as np
import matplotlib.pyplot as plt
from projects.phi.tools.utils import *
from scipy.stats import mannwhitneyu
from scipy.stats import entropy
import numpy as np
import networkx as nx
from projects.generalize_ising_model.core import generalized_ising
from projects.generalize_ising_model.tools.utils import to_normalize, to_save_results, correlation_function, dim, find_nearest

hubs = np.squeeze(load_matrix('/home/user/Desktop/data_phi/tpm/DMN/Hubs.npy'))

print(hubs.shape)


for sub in range(hubs.shape[1]):
    #minHub,maxHub = [],[]
    hub_count1, hub_count2,hub_count3,hub_count4,hub_count5 = [],[],[],[],[]
    for cond in range(hubs.shape[0]):
        minHub = (np.min(hubs[cond,sub,:]))
        maxHub = (np.max(hubs[cond,sub,:]))

        quart = (maxHub-minHub)/4

        fourth_quartile =  3*quart + minHub
        third_quartile = 2*quart + minHub


        hub_count1.append(np.squeeze(np.where(hubs[cond,sub,:]>fourth_quartile)).shape)
        hub_count2.append(np.squeeze(np.where(hubs[cond, sub, :] > third_quartile)).shape)
        hub_count3.append(np.squeeze(np.where(hubs[cond, sub, :] > 1.5*minHub)).shape)
        hub_count4.append(np.squeeze(np.where(hubs[cond, sub, :] > 3 * minHub)).shape)
        hub_count5.append(np.squeeze(np.where(hubs[cond, sub, :] > 2 * minHub)).shape)


    plt.scatter(np.arange(len(hub_count1)), hub_count1)
    plt.scatter(np.arange(len(hub_count2)), hub_count2)
    plt.scatter(np.arange(len(hub_count3)), hub_count3)
    plt.scatter(np.arange(len(hub_count4)), hub_count4)
    plt.scatter(np.arange(len(hub_count5)), hub_count5)
    plt.legend()
    plt.show()

