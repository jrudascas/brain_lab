from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_probability(wavefun):

    amplitude = np.abs(wavefun).T

    ampMag = np.sqrt(np.sum((amplitude * amplitude).T, axis=0))

    normAmp = (np.asarray(amplitude.T) / np.asarray(ampMag)).T

    return normAmp * normAmp


main_path = '/home/user/Desktop/QMBrain/New Data/'

filepathX = main_path + 'x_chanloc.csv'
filepathY = main_path + 'y_chanloc.csv'

x = load_matrix(filepathX)
y = load_matrix(filepathY)


byd_x = load_matrix('/home/user/Desktop/QMBrain/BYD/Cond10/xAvg_time.npy')
byd_y = load_matrix('/home/user/Desktop/QMBrain/BYD/Cond10/yAvg_time.npy')

byd_s_x = load_matrix('/home/user/Desktop/QMBrain/BYD/Cond12/xAvg_time.npy')
byd_s_y = load_matrix('/home/user/Desktop/QMBrain/BYD/Cond12/yAvg_time.npy')

taken_x = load_matrix('/home/user/Desktop/QMBrain/New Data/Cond10/xAvg_time.npy')
taken_y = load_matrix('/home/user/Desktop/QMBrain/New Data/Cond10/yAvg_time.npy')

taken_s_x = load_matrix('/home/user/Desktop/QMBrain/New Data/Cond12/xAvg_time.npy')
taken_s_y = load_matrix('/home/user/Desktop/QMBrain/New Data/Cond12/yAvg_time.npy')

rest_x = load_matrix('/home/user/Desktop/QMBrain/RestData/xAvg_time.npy')
rest_y = load_matrix('/home/user/Desktop/QMBrain/RestData/yAvg_time.npy')


movie_x = load_matrix('/home/user/Desktop/QMBrain/new_movie/xAvg_time.npy')
movie_y = load_matrix('/home/user/Desktop/QMBrain/new_movie/yAvg_time.npy')

print('BYD X',byd_x)
print('BYD y',byd_y)

print('BYD s X',byd_s_x)
print('BYD s y',byd_s_y)

print('t X',taken_x)
print('t y',taken_y)

print('t s X',taken_s_x)
print('t s y',taken_s_y)

print('r X',rest_x)
print('r y',rest_y)

print('m X',movie_x)
print('m y',movie_y)

plt.scatter(x,y,color='k')

plt.scatter(byd_x,byd_y)
plt.scatter(byd_s_x,byd_s_y)
plt.scatter(taken_x,taken_y)
plt.scatter(taken_s_x,taken_s_y)
plt.scatter(rest_x,rest_y)
plt.scatter(movie_x,movie_y)

plt.legend

plt.show()