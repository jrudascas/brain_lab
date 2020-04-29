
from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib

matplotlib.rcParams.update({'font.size': 18})

stimuli =[('New Data/',15),('BYD/',14),('nyc_data/',13)]


titles_raw = ['Taken Raw','Taken Scrambled Raw','BYD Raw','BYD Scrambled Raw','Rest Raw','Present Raw']
titles_prob = ['Taken Probability','Taken Scrambled Probability','BYD Probability','BYD Scrambled Probability','Rest Probability','Present Probability']
titles_phase = ['Taken Phase','Taken Scrambled Phase','BYD Phase','BYD Scrambled Phase','Rest Phase','Present Phase']

count=0

raws,probs,phases = [],[],[]

for stimulus in stimuli:

    main_path = '/home/user/Desktop/QMBrain/' + stimulus[0]

    x = load_matrix(main_path + 'x_chanloc.csv')
    y = load_matrix(main_path + 'y_chanloc.csv')


    condition_list = ['Cond10/','Cond12/']

    num_sub = stimulus[1]
    num_electrodes = 92

    list_sub = np.arange(num_sub)



    for condition in condition_list:


        load_path = main_path + condition +'/'

        isc_avg = load_matrix(load_path+'avg_isc.csv')
        isc_avg_prob = load_matrix(load_path+'avg_isc_prob.csv')
        isc_avg_phase = load_matrix(load_path+'avg_isc_phase.csv')

        df = pd.DataFrame({'x':x,'y':y,'isc_avg':isc_avg,'isc_prob':isc_avg_prob,'isc_phase':isc_avg_phase})

        norm = plt.Normalize(-0.007,0.05)
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])

        fig = plt.figure(figsize=(16,9))
        ax = sns.scatterplot(x="x", y="y",
                             hue="isc_avg", size="isc_avg",
                             sizes=(20, 200),palette='plasma', hue_norm=(-0.007,0.05),
                             data=df,legend=False)
        plt.colorbar(sm)
        plt.suptitle(titles_raw[count],fontsize=18)
        #plt.show()
        plt.savefig(main_path+condition+'isc_map.png',dpi=600)
        plt.close()


        norm_prob = plt.Normalize(-0.0075, 0.08)
        sm_prob = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm_prob.set_array([])

        fig1 = plt.figure(figsize=(16, 9))
        ax1 = sns.scatterplot(x="x", y="y",
                             hue="isc_prob", size="isc_prob",
                             sizes=(20, 200), palette='plasma', hue_norm=(-0.0075, 0.08),
                             data=df, legend=False)
        plt.colorbar(sm_prob)
        plt.suptitle(titles_prob[count], fontsize=18)
        #plt.show()
        plt.savefig(main_path + condition + 'prob_map.png', dpi=600)
        plt.close()


        norm_phase = plt.Normalize(0.5, 1)
        sm_phase = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm_phase.set_array([])

        fig2 = plt.figure(figsize=(16, 9))
        ax2 = sns.scatterplot(x="x", y="y",
                             hue="isc_phase", size="isc_phase",
                             sizes=(20, 200), palette='plasma', hue_norm=(0.5, 1),
                             data=df, legend=False)
        plt.colorbar(sm_phase)
        plt.suptitle(titles_phase[count], fontsize=18)
        #plt.show()
        plt.savefig(main_path + condition + 'phase_map.png', dpi=600)
        plt.close()

        count+=1



'''

fig = plt.figure(figsize=(20, 10), dpi=600)
plt.scatter(x, y, alpha=0.4,c=isc_avg,vmin=-0.00005,vmax=0.001)
plt.xlim(-10, 10)
plt.ylim(-8, 8)
plt.colorbar()
plt.title('Raw')
plt.savefig(main_path+condition+'raw_map.png',dpi=600)
plt.close(fig)

fig2 = plt.figure(figsize=(20, 10), dpi=600)
plt.scatter(x, y, alpha=0.4, c=isc_avg_prob,vmin=-0.0004,vmax=0.001)
plt.xlim(-10, 10)
plt.ylim(-8, 8)
plt.colorbar()
plt.title('Prob')
plt.savefig(main_path+condition+'prob_map.png',dpi=600)
plt.close(fig2)

fig3 = plt.figure(figsize=(20, 10), dpi=600)
plt.scatter(x, y, alpha=0.4, c=isc_avg_phase,vmin=0.01,vmax=0.026)
plt.xlim(-10, 10)
plt.ylim(-8, 8)
plt.colorbar()
plt.title('Phase')
plt.savefig(main_path + condition + 'phase_map.png', dpi=600)
plt.close(fig3)


'''
