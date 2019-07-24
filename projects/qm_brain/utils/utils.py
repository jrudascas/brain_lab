import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from scipy.signal import hilbert
from pynufft import NUFFT_cpu

def load_matrix(filepath):
    extension = filepath.split('.')[-1]
    if str(extension) == 'csv':
        return np.genfromtxt(filepath,delimiter = ',')
    elif str(extension) == 'npy':
        return np.load(filepath)
    elif str(extension) == 'mat':
        return scipy.io.loadmat(filepath)
    elif str(extension) == 'npz':
        return np.load(filepath)

def file_exists(filename):
    import os
    exists = os.path.isfile(filename)
    if exists:
        return True
    else:
        return False

def makedir2(path):
    import os
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return True

def hilberto(data):
    assert len(data.shape) == 2
    if data.shape[1]>data.shape[0]:
        return hilbert(data,axis=1)
    else:
        return hilbert(data,axis=0).T

def process_eeg_data(data):


    hilbertTransData = hilberto(data)

    amplitude = np.abs(hilbertTransData).T


    phase = np.unwrap(np.angle(hilbertTransData)).T

    ampMag = np.sqrt(np.sum((amplitude * amplitude).T, axis=0))

    normAmp = (np.asarray(amplitude.T) / np.asarray(ampMag)).T

    probability = normAmp * normAmp

    del ampMag, amplitude, hilbertTransData

    return phase, normAmp, probability

def process_eeg_data2(data):


    hilbertTransData = hilberto(data)

    amplitude = np.abs(hilbertTransData).T

    ampMag = np.sqrt(np.sum((amplitude * amplitude).T, axis=0))

    normAmp = (np.asarray(amplitude.T) / np.asarray(ampMag)).T

    probability = normAmp * normAmp

    del ampMag, amplitude,normAmp

    return probability, hilbertTransData

def animation_station2(xAvg,yAvg,xInit,yInit):
    fig,ax = plt.subplots(figsize=(10, 6))
    ax.set(xlim=(-10,10),ylim=(-8,8))

    scat = ax.scatter(xAvg,yAvg)

    # initialization function: plot the background of each frame
    def init():
        ax.set_data(xInit, yInit)
        return scat,

    # animation function.  This is called sequentially
    def animate(i):
        # fig.errorbar(xAvg[i], yAvg[i], dy, dy, dx, dx, marker='.')
        x1 = xAvg[i]
        y1 = yAvg[i]
        scat.set_data(x1, y1)
        return scat,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = FuncAnimation(fig, animate, init_func=init,
                                   frames=300, interval=20)  # , blit=True)

    anim.save('animationTest.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

def animate_anigreat(max_ind,xIn,yIn,sub_num,cond,numtime=300):
    plt.close()
    for i in range(numtime):
        fig = plt.figure(figsize=(480/96, 480 / 96), dpi=96)
        plt.scatter(xIn,yIn,alpha=0.4)
        plt.scatter(xIn[max_ind[i]], yIn[max_ind[i]],edgecolors="grey", linewidth=2)
        plt.xlim(-10, 10)
        plt.ylim(-8, 8)
        filename = 'step' + str(i) + '.png'
        plt.savefig(filename, dpi=96)
        plt.gca()
        plt.close(fig)
    bashCommand = "convert -delay 10 *.png "+ str(sub_num)+"_"+str(cond)+"_fig.gif"
    os.system(bashCommand)


def animation_station(xAvg,yAvg,xIn,yIn,xerr,yerr):
    for i in range(300):
        fig = plt.figure(figsize=(480/96, 480 / 96), dpi=96)
        plt.scatter(xIn,yIn)
        plt.scatter(xAvg[i], yAvg[i],edgecolors="grey", linewidth=2)
        plt.xlim(-10, 10)
        plt.ylim(-8, 8)
        filename = 'step' + str(i) + '.png'
        plt.savefig(filename, dpi=96)
        plt.errorbar(xerr[i],yerr[i])
        plt.gca()
        plt.close(fig)
    bashCommand = "convert -delay 80 *.png animated_chart_momentum_nick.gif"
    os.system(bashCommand)

def probability_conservation_plot(num_electrodes,data):
    y = []
    x=np.linspace(0,num_electrodes,num=num_electrodes)
    for i in range(num_electrodes):
        y.append(np.sum(data[i,:]))
    plt.plot(x,y,color='r')
    plt.show()

def momentum_from_position(avg_position,ind=1):
    momentum = []
    for i in range(avg_position.shape[-1]):
        if i == (avg_position.shape[-1]-1):
            position = avg_position[i]
            momentum.append(position/ind)
        else:
            position = avg_position[i + ind] - avg_position[i]
            momentum.append(position / ind)
    return np.asarray(momentum)

def momentum_wavefunc(pos_wavefunc,norm='ortho',axis=1):
    momentum_wavefunction = np.fft.fft(pos_wavefunc,norm=norm,axis=axis)
    return momentum_wavefunction

def momenta_prob(momentum_wavefunction):

    pAmp = np.abs(momentum_wavefunction)

    pPhase = np.unwrap(np.angle(momentum_wavefunction))

    ampMag = np.sqrt(np.sum((pAmp * pAmp).T, axis=0))

    normpAmp = (np.asarray(pAmp.T) / np.asarray(ampMag)).T

    momentum_prob = normpAmp * normpAmp

    del ampMag, pAmp

    return pPhase, normpAmp, momentum_prob

def plot_avg(avg1,avg2,times,ylabel = 'Position (cm)',title = 'Average Position as Function of Time',path_output = None):
    f = plt.figure()
    plt.plot(times, avg1)
    plt.plot(times, avg2)
    plt.title(title)
    plt.legend('x', 'y')
    plt.xlabel('Time (microseconds)')
    plt.ylabel(ylabel)
    plt.xlim([0, 1000])
    plt.ylim(([-6, 6]))
    if path_output is not None:
        plt.savefig(path_output, '/' + 'plot.png', dpi=1200)
        plt.close()

def prob_derivative(probability,ind=1):
    p = []
    for i in range(probability.shape[0]):
        if i == (probability.shape[0]-1):
            p_i = probability[i,:]
            p.append(p_i/ind)
        else:
            p_i = probability[i + ind,:] - probability[i,:]
            p.append(p_i / ind)
    return np.asarray(p)

def save_file(data,path,name):

    default_delimiter = ','
    format = '%1.5f'

    if len(data.shape) <= 2:
        file = path + str(name) + '.csv'

        if not file_exists(file):
            np.savetxt(file, np.asarray(data), delimiter=default_delimiter, fmt=format)
    else:
        file = path + str(name) + '.npy'
        if not file_exists(file):
            np.save(file,np.asarray([data]))


def save_tpm(tpm,path_output,num):
    import numpy as np
    from nilearn import plotting
    import matplotlib.pyplot as plt
    path_output = path_output + '/data'
    if makedir2(path_output):
        default_delimiter = ','
        format = '%1.5f'
        filename1 = path_output + '/' + 'tpm_'+ str(num) + '.csv'

        if not file_exists(filename1):
            np.savetxt(filename1, tpm, delimiter=default_delimiter, fmt=format)

        fig, ax = plt.subplots()

        plotting.plot_matrix(tpm, figure=fig, vmax=1, vmin=0)

        filename2 = path_output + '/' + 'plots_' + str(num)+'.png'
        if not file_exists(filename2):
            fig.savefig(filename2, dpi=1200)

        plt.close(fig)

def zip_x_y(x,y):
    assert len(x) == len(y)
    return np.stack((x,y))


def data_1d_to_2d(data,x,y):

    # Get the indices of the original x-positions that lead to a sorted (- to +) vector
    x_inds = np.argsort(x)

    # Actually sort the y vector from (- to +)
    y_sorted = np.sort(y)

    flag = False

    assert len(data.shape) == 2
    if data.shape[1] > data.shape[0]:
        flag = True
        data = data.T

    new_dat = np.zeros(shape=(len(y), len(x), data.shape[0]),dtype=np.complex64)

    y_count = 0

    for count, x_ind in enumerate(x_inds):

        y_loc = np.where(y_sorted == y[x_ind])

        if np.squeeze(np.squeeze(y_loc).shape) > 1:
            y_loc = np.squeeze(y_loc)[y_count]

            y_count += 1

        new_dat[y_loc, count, :] = data[:, x_ind]

    if flag is True:
        new_dat = new_dat.T

    del x_inds, y_sorted, flag, data, y_count,y_loc, count, x_ind

    return new_dat

def data_2d_to_1d(data,x,y):
    # Get the indices of the original x-positions that lead to a sorted (- to +) vector
    x_inds = np.argsort(x)

    # Actually sort the y vector from (- to +)
    y_sorted = np.sort(y)


    size = data.shape

    assert len(size) == 3
    assert size[2] == size[1]
    assert size[0] > size[1]

    new_dat = np.zeros(shape=(size[0],size[1]))

    y_count = 0

    for count, x_ind in enumerate(x_inds):

        y_loc = np.where(y_sorted == y[x_ind])

        if np.squeeze(np.squeeze(y_loc).shape) > 1:
            y_loc = np.squeeze(y_loc)[y_count]

            y_count += 1

        new_dat[:,count] = data[:,y_loc,x_ind]

    return new_dat



def non_uniform_fft(pos_stack,pos_wavefun,solver,interp_size):

    assert len(pos_wavefun.shape) == 2


    NufftObj = NUFFT_cpu()

    om = pos_stack
    Nd = (len(pos_stack[0]),len(pos_stack[1]))
    Kd = Nd
    Jd = (interp_size,interp_size)

    NufftObj.plan(om,Nd,Kd,Jd)

    y = NufftObj.forward(pos_wavefun)

    mom_wavefun_1 = NufftObj.solve(y,solver=solver )

    #mom_wavefun_2 = NufftObj.adjoint(y)

    return mom_wavefun_1 #, mom_wavefun_2


def fft_time_warp(pos_stack, pos_wavefun,solver='cg',interp_size = 8):
    size = pos_wavefun.shape

    assert len(size) == 3
    assert size[2] == size[1]
    assert size[0] > size[1]


    momenta_wavefun = np.zeros(shape=(size[0], size[1], size[2]),dtype=np.complex64)

    for t in range(pos_wavefun.shape[0]):
        momenta_wavefun[t,...] = non_uniform_fft(pos_stack,pos_wavefun[t,...],solver,interp_size)

    return momenta_wavefun

