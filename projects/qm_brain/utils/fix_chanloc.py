import numpy as np
from projects.qm_brain.utils.utils import load_matrix, save_file


chanLocs = load_matrix('/home/user/Desktop/QMBrain/EEG Data/1/chanLocXY.csv')
x = chanLocs[:, 0]
y = chanLocs[:, 1]

x = x[:-1]
y = y[:-1]

allchan = np.arange(128)

chanexcl = np.array([1,8,14,17,21,25,32,38,43,44,48,49,56,63,64,68,69,73,74,81,82,88,89,94,95,99,107,113,114,119,120,121,125,126,127,128])-1


incchn = np.setdiff1d(allchan,chanexcl)

new_x = x[incchn]
new_y = y[incchn]

path = '/home/user/Desktop/QMBrain/New Data/'

save_file(new_x,path,'x_chanloc2')
save_file(new_y,path,'y_chanloc2')

data = load_matrix('/home/user/Desktop/QMBrain/New Data/Cond12/1/data.csv')

times = [0,]

for i in range(data.shape[0]-1):
    times.append(times[i]+4)

save_file(np.asarray(times),path,'times')