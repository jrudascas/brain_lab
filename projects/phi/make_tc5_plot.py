from projects.phi.tools.utils import make_ts_array,load_matrix
import numpy as np
import matplotlib.pyplot as plt

ts_path = '/home/user/Desktop/data_phi/Propofol/Awake/datasink/preprocessing/sub-2014_05_16_16RA/_image_parcellation_path_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..Default..Default_parcellation_5.nii/time_series.csv'

ts = load_matrix(ts_path)

ts2 = np.copy(ts)

print(ts.shape)

ts1_other = np.copy(ts[:,0])
ts2_other = np.copy(ts[:,1])
ts3_other = np.copy(ts[:,2])
ts4_other = np.copy(ts[:,3])
ts5_other = np.copy(ts[:,4])

ts2[np.where(ts1_other >= np.mean(ts[:,0])),0] = 1
ts2[np.where(ts1_other < np.mean(ts[:,0])),0] = 0

ts2[np.where(ts2_other >= np.mean(ts[:,1])),1] = 1
ts2[np.where(ts2_other < np.mean(ts[:,1])),1] = 0

ts2[np.where(ts3_other >= np.mean(ts[:,2])),2] = 1
ts2[np.where(ts3_other < np.mean(ts[:,2])),2] = 0

ts2[np.where(ts4_other >= np.mean(ts[:,3])),3] = 1
ts2[np.where(ts4_other < np.mean(ts[:,3])),3] = 0

ts2[np.where(ts5_other >= np.mean(ts[:,4])),4] = 1
ts2[np.where(ts5_other < np.mean(ts[:,4])),4] = 0


fig = plt.figure()
ax1 = fig.add_subplot(5, 1, 1)
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
plt.xticks([])
plt.yticks([])
ax1.set_ylim([-5, 5])
plt.plot(ts[:,0])
ax1.fill_between(np.arange(245),-5,5,where=(ts2[:,0]==1),color='r',alpha=0.3)
plt.axhline(np.mean(ts[:,0]))

ax2 = fig.add_subplot(5, 1, 2)
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
plt.xticks([])
plt.yticks([])
ax2.set_ylim([-5, 5])
plt.plot(ts[:,1])
ax2.fill_between(np.arange(245),-5,5,where=(ts2[:,1]==1),color='r',alpha=0.3)
plt.axhline(np.mean(ts[:,1]))

ax3 = fig.add_subplot(5, 1, 3)
ax3.spines['right'].set_color('none')
ax3.spines['top'].set_color('none')
plt.xticks([])
plt.yticks([])
ax3.set_ylim([-5, 5])
plt.plot(ts[:,2])
ax3.fill_between(np.arange(245),-5,5,where=(ts2[:,2]==1),color='r',alpha=0.3)
plt.axhline(np.mean(ts[:,2]))

ax4 = fig.add_subplot(5, 1, 4)
ax4.spines['right'].set_color('none')
ax4.spines['top'].set_color('none')
plt.xticks([])
plt.yticks([])
ax4.set_ylim([-5, 5])
plt.plot(ts[:,3])
ax4.fill_between(np.arange(245),-5,5,where=(ts2[:,3]==1),color='r',alpha=0.3)
plt.axhline(np.mean(ts[:,3]))

ax5 = fig.add_subplot(5, 1, 5)
ax5.spines['right'].set_color('none')
ax5.spines['top'].set_color('none')
plt.xticks([])
plt.yticks([])
ax5.set_ylim([-5, 5])
plt.plot(ts[:,4])
ax5.fill_between(np.arange(245),-5,5,where=(ts2[:,4]==1),color='r',alpha=0.3)
plt.axhline(np.mean(ts[:,4]))

plt.show()


