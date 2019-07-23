from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
main_path = '/home/user/Desktop/QMBrain/New Data/'

filepathX = main_path + 'x_chanloc.csv'
filepathY = main_path + 'y_chanloc.csv'

x = load_matrix(filepathX)
y = load_matrix(filepathY)

xx,yy = np.meshgrid(x,y)

coord_stack = np.stack((x,y))

# Sort the vectors and keep previous indices

x_inds = np.argsort(x)

y_sorted = np.sort(y)


condition_list = ['Cond10/','Cond12/']

for condition in condition_list:

    for i in range(3):

        subject_path = main_path + condition + str(i + 1) + '/'

        if file_exists(subject_path + 'DeltaX.csv'):

            print('Running for subject ', i+1, 'in folder ', condition)

            filepathData = subject_path + 'data.csv'

            data = load_matrix(filepathData)

            new_dat = np.zeros(shape=(len(y),len(x),data.shape[0]))

            for count, x_ind in enumerate(x_inds):

                y_loc = np.where(y_sorted == y[x_ind])

                if len(np.squeeze(y_loc).shape) > 1:
                    y_loc_2 = np.where(coord_stack[0,:] == x[x_ind] and coord_stack[1,:] == y[x_ind])

                    y_loc = np.setdiff1d(y_loc,y_loc_2)

                new_dat[y_loc,count,:] = data[:,x_ind]

            print(new_dat.shape)

            plt.scatter(x[x_inds],y[x_inds])

