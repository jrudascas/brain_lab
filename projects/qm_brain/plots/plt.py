from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt


main_path = '/home/user/Desktop/QMBrain/RestData/'

filepathX = main_path + 'x_chanloc.csv'
filepathY = main_path + 'y_chanloc.csv'

x = load_matrix(filepathX)
y = load_matrix(filepathY)

coord_stack = zip_x_y(x,y)

condition_list = ['/']#'Cond10/','Cond12/']


for condition in condition_list:

    dx_list, dp_list = [],[]

    for i in range(13):

        subject_path = main_path + condition + str(i + 1) + '/results/'

        print('Running for subject ', i + 1, 'in folder ', condition)


        filepathPos = subject_path + 'DeltaX.csv'
        filepathMom = subject_path + 'DeltaPX.csv'

        dx_list.append(load_matrix(filepathPos))
        dp_list.append(load_matrix(filepathMom))

    plt.figure()
    plt.plot(dx_list)
    #plt.plot(dp_list,color='k')
    plt.xlim([0,200])
    plt.show()

