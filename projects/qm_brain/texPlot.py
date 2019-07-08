from projects.qm_brain.utils import  *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


times= load_matrix('/home/user/Desktop/QMBrain/New Data/times.csv')

uncertainity_x1 = load_matrix('/home/user/Desktop/QMBrain/New Data/Cond10/1/DeltaXDeltaPX.csv')
uncertainity_y1 = load_matrix('/home/user/Desktop/QMBrain/New Data/Cond10/1/DeltaYDeltaPY.csv')

uncertainity_x2 = load_matrix('/home/user/Desktop/QMBrain/New Data/Cond10/2/DeltaXDeltaPX.csv')
uncertainity_y2 = load_matrix('/home/user/Desktop/QMBrain/New Data/Cond10/2/DeltaYDeltaPY.csv')


uncertainity_x3 = load_matrix('/home/user/Desktop/QMBrain/New Data/Cond10/3/DeltaXDeltaPX.csv')
uncertainity_y3 = load_matrix('/home/user/Desktop/QMBrain/New Data/Cond10/3/DeltaYDeltaPY.csv')


uncertainity_x4 = load_matrix('/home/user/Desktop/QMBrain/New Data/Cond10/4/DeltaXDeltaPX.csv')
uncertainity_y4 = load_matrix('/home/user/Desktop/QMBrain/New Data/Cond10/4/DeltaYDeltaPY.csv')


plt.clf()
plt.close()
fig = plt.figure()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.9)

fig.add_subplot(2,2,1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(times, uncertainity_x1)
#plt.ylim([0,8])

fig.add_subplot(2,2,2)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(times, uncertainity_x2)
#plt.ylim([0,8])

fig.add_subplot(2,2,3)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(times, uncertainity_x3)
#plt.ylim([0,8])

fig.add_subplot(2,2,4)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(times, uncertainity_x4)
#plt.ylim([0,8])



fig.text(0.5, 0.04, r'\textbf{Time} $(\mu s)$', ha='center')
fig.text(0.01, 0.5, r'$\Delta x(t) \Delta p_x(t)$', va='center', rotation='vertical')


plt.suptitle(r'\textbf{Uncertainty Relation in EEG Data (x)}',fontsize=16)

plt.savefig('uncX')
plt.show()