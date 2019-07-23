from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from scipy import sparse
main_path = '/home/user/Desktop/QMBrain/New Data/'

filepathX = main_path + 'x_chanloc.csv'
filepathY = main_path + 'y_chanloc.csv'

x = load_matrix(filepathX)
y = load_matrix(filepathY)

x_inds = np.argsort(x)
y_inds = np.argsort(y)

#grid_x, grid_y = np.mgrid[min(x):max(x):300j,min(y):max(y):200j]

condition_list = ['Cond10/','Cond12/']

dim = np.stack((x_inds,y_inds))

for condition in condition_list:

    for i in range(3):

        subject_path = main_path + condition + str(i + 1) + '/'

        if file_exists(subject_path + 'DeltaX.csv'):

            print('Running for subject ', i+1, 'in folder ', condition)

            filepathData = subject_path + 'data.csv'

            data = load_matrix(filepathData)
            new_dat = np.zeros(shape=(data.shape[1],data.shape[1],data.shape[0]))
            #new_dat = griddata(np.stack((x,y)),data,(x,y))
            for t in range(data.shape[0]):
                new_dat[...,t] = sparse.coo_matrix((data[t,:],dim),(data.shape[1],data.shape[1])).A

            print(new_dat.shape)

