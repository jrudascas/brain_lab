from projects.qm_brain.utils.utils import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


'''
def get_data(table,rownum,title):
    data = pd.DataFrame(table.loc[rownum][2:]).astype(float)
    data.columns = {title}
    return data

num_subjects = 15
main_path = '/home/user/Desktop/QMBrain/New Data/'

filepathX = main_path + 'x_chanloc2.csv'
filepathY = main_path + 'y_chanloc2.csv'
filepathTimes = main_path + 'times.csv'

times = load_matrix(filepathTimes)
x = load_matrix(filepathX)
y = load_matrix(filepathY)

filepathData = main_path + 'Cond10/1/data.csv'

data = load_matrix(filepathData)

phase, normAmp, probability = process_eeg_data(data)

max_inds = np.argmax(probability, axis=1)

psi = normAmp * np.exp(1j * phase)

xAvg = probability @ x
yAvg = probability @ y

v_x_avg = np.diff(xAvg)
v_y_avg = np.diff(yAvg)
v_tot = np.sqrt((v_y_avg**2 + v_x_avg**2))

norm_v = v_tot/np.max(v_tot)

plt.close()
plt.scatter(x,y,color='k')
plt.scatter(xAvg[:76749], yAvg[:76749], linewidth=2,c=norm_v,cmap='viridis')
plt.colorbar()
plt.xlim(-10, 10)
plt.ylim(-8, 8)
plt.show()

'''

#animate_anigreat_w_velocity(xAvg[:300],yAvg[:300], x, y, v_tot[:300],sub_num=1, cond='fig')



def main():
    numframes = 100
    numpoints = 10
    main_path = '/home/user/Desktop/QMBrain/New Data/'

    filepathX = main_path + 'x_chanloc2.csv'
    filepathY = main_path + 'y_chanloc2.csv'
    filepathTimes = main_path + 'times.csv'

    x = load_matrix(filepathX)
    y = load_matrix(filepathY)
    color_data = np.random.random((numframes, numpoints))

    xIn = x
    yIn = y

    times = load_matrix(filepathTimes)

    filepathData = main_path + 'Cond10/1/data.csv'

    data = load_matrix(filepathData)

    phase, normAmp, probability = process_eeg_data(data)

    max_inds = np.argmax(probability, axis=1)

    psi = normAmp * np.exp(1j * phase)

    xAvg = probability @ x
    yAvg = probability @ y



    v_x_avg = np.diff(xAvg)
    v_y_avg = np.diff(yAvg)
    v_tot = np.sqrt((v_y_avg ** 2 + v_x_avg ** 2))

    norm_v = v_tot / np.max(v_tot)


    c = norm_v

    fig = plt.figure()
    plt.scatter(xIn,yIn,color='k')
    scat = [plt.scatter(xAvg[:i], yAvg[:i], c=c[:i], s=100) for i in range(76749)]

    ani = animation.FuncAnimation(fig, update_plot, frames=range(numframes),
                                  fargs=(color_data, scat))
    plt.show()

def update_plot(i, data, scat):
    scat.set_array(data[i])
    return scat,

#main()

main_path = '/home/user/Desktop/QMBrain/New Data/'

filepathX = main_path + 'x_chanloc2.csv'
filepathY = main_path + 'y_chanloc2.csv'
filepathTimes = main_path + 'times.csv'

x = load_matrix(filepathX)
y = load_matrix(filepathY)


xIn = x
yIn = y

times = load_matrix(filepathTimes)

filepathData = main_path + 'Cond10/1/data.csv'

data = load_matrix(filepathData)

phase, normAmp, probability = process_eeg_data(data)

max_inds = np.argmax(probability, axis=1)

psi = normAmp * np.exp(1j * phase)

xAvg = probability @ x
yAvg = probability @ y

xAvg = xAvg[:300]
yAvg = yAvg[:300]

v_x_avg = np.diff(xAvg)
v_y_avg = np.diff(yAvg)
v_tot = np.sqrt((v_y_avg ** 2 + v_x_avg ** 2))

norm_v = v_tot / np.max(v_tot)

c = norm_v[:300]

norm = plt.Normalize(min(norm_v), max(norm_v))
sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
sm.set_array([])

for i in range(len(c)):
    fig = plt.figure()
    plt.scatter(xIn, yIn, color='k')
    plt.scatter(xAvg[i], yAvg[i], color = sm.to_rgba(c[i]), cmap='plasma')
    plt.colorbar(sm)
    plt.xlim(-10, 10)
    plt.ylim(-8, 8)
    filename = 'step' + str(i) + '.png'
    plt.savefig(filename, dpi=96)
    plt.gca()
    plt.close(fig)
bashCommand = "convert -delay 40 *.png "  + "new_fig.gif"
os.system(bashCommand)

