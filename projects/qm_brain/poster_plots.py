from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
import pandas as pd

# Define a palette to ensure that colors will be
# shared across the facets




main_path = '/home/user/Desktop/QMBrain/New Data/'

filepathX = main_path + 'x_chanloc.csv'
filepathY = main_path + 'y_chanloc.csv'

x = norm_on_int_pi(load_matrix(filepathX))
y = norm_on_int_pi(load_matrix(filepathY))

coord_stack = zip_x_y(x,y)

condition_list = ['Cond10/','Cond12/']



for condition in condition_list:

    for i in range(14):
        subject_path = main_path + condition + str(i + 1) + '/'
        filepathData = subject_path + 'data.csv'

        data = load_matrix(filepathData)

        phase,normamp,probability = process_eeg_data(data)

        wavefun = normamp*np.exp(1j*phase)

        plt.plot(data[:1000,0],color = 'r',alpha=0.5)
        plt.plot(wavefun[:1000, 0], color='b', alpha=0.5)
        plt.show()

        d = {'Data':np.stack((data[:1000,0],wavefun[:1000,0])),'type':np.stack((str(np.ones(1000)),str(np.zeros(1000)))),'Time':np.stack((np.arange(1000),np.arange(1000)))}

        df = pd.DataFrame(data=d)

        palette = dict(zip(df.type.unique(),
                           sns.color_palette("rocket_r", 6)))

        #plt.plot(data[:1000, 0])
       # plt.plot(wavefun[:1000, 0])


        #plt.show()
        # Plot the lines on two facets
        sns.relplot(x="Time", y="Data",
                    hue="type",
                    size_order=[1, 0], palette=palette,
                    height=5, aspect=.75, facet_kws=dict(sharex=False),
                    kind="line", legend="full", data=df)

        f = 1
