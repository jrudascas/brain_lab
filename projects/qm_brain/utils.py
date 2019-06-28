import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from scipy.signal import hilbert

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