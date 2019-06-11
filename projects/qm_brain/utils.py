import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from scipy.signal import hilbert

def load_matrix(filepath):
    extension = filepath.split('.')[1]
    if str(extension) == 'csv':
        return np.genfromtxt(filepath,delimiter = ',')
    elif str(extension) == 'npy':
        return np.load(filepath)
    elif str(extension) == 'mat':
        return scipy.io.loadmat(filepath)
    elif str(extension) == 'npz':
        return np.load(filepath)

def process_eeg_csv(data,chanLocs):
    x = chanLocs[:, 0]
    y = chanLocs[:, 1]


    hilbertTransData = hilbert(data, axis=1)

    amplitude = np.abs(hilbertTransData).T

    wavefunction = hilbertTransData

    phase = np.unwrap(np.angle(hilbertTransData)).T

    ampMag = np.sqrt(np.sum((amplitude * amplitude).T, axis=0))

    normAmp = (np.asarray(amplitude.T) / np.asarray(ampMag)).T

    probability = normAmp * normAmp

    del ampMag, amplitude, hilbertTransData, chanLocs

    return x, y, wavefunction, phase, normAmp, probability


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

def momentum_wavefunction(pos_wavefunc,norm='ortho',axis=1):
    momentum_wavefunction = np.fft.fft(pos_wavefunc,norm=norm,axis=axis)
    #p_y = np.fft.fft(y_wavefunc, norm=norm, axis=axis)
    return momentum_wavefunction