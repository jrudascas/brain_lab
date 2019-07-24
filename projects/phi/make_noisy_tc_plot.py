from projects.phi.tools.utils import make_ts_array,load_matrix
import numpy as np
import matplotlib.pyplot as plt

ts_path = '/home/user/Desktop/data_phi/Propofol/Awake/datasink/preprocessing/sub-2014_05_16_16RA/_image_parcellation_path_..home..brainlab..Desktop..Rudas..Data..Parcellation..rsn_parcellations..Default..Default_parcellation_5.nii/time_series.csv'

ts = load_matrix(ts_path)

print(ts.shape)

mean = np.mean(ts[:,0])

plt.xkcd()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xticks([])
plt.yticks([])
ax.set_ylim([-15, 15])


plt.annotate(
    'Above the Mean\nEquals 1',
    xy=(103, 2), arrowprops=dict(arrowstyle='->'), xytext=(15, 11))

plt.annotate('Below the Mean\nEquals 0',xy=(162, -1.3), arrowprops=dict(arrowstyle='->'), xytext=(133, -10))

plt.plot(ts[:,0])
plt.axhline(mean)

plt.xlabel('Time')

plt.show()

fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows = 5, ncols=1)

ax1.xticks([])
ax1.yticks([])
ax1.plot(ts[:,0])
ax1.axhline(np.mean(ts[:,0]))


ax2.xticks([])
ax2.yticks([])

ax2.plot(ts[:,1])
ax2.axhline(np.mean(ts[:,1]))


ax3.xticks([])
ax3.yticks([])

ax3.plot(ts[:,2])
ax3.axhline(np.mean(ts[:,2]))


ax4.xticks([])
ax4.yticks([])

ax4.plot(ts[:,3])
ax4.axhline(np.mean(ts[:,3]))


ax5.xticks([])
ax5.yticks([])
ax5.plot(ts[:,4])
ax5.axhline(np.mean(ts[:,4]))

plt.show(fig)


time_series_other = np.copy(ts[:,0])

ts[np.where(time_series_other >= mean),0] = 0
ts[np.where(time_series_other < mean),0] = 1

plt.scatter(np.arange(len(ts[:,0])),ts[:,0])
plt.plot(ts[:,0])
plt.show()



