import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def load_matrix(file):
    extension = file.split('.')[1]
    if str(extension) == 'csv':
        return np.genfromtxt(file,delimiter = ',')
    elif str(extension) == 'npy':
        return np.load(file)
    elif str(extension) == 'mat':
        return scipy.io.loadmat(file)
    elif str(extension) == 'npz':
        return np.load(file)

def moving_avg(array):
    return np.sum(array,axis=0)/array.shape[-1]

def moving_std(array):
    return np.std(array,axis=0)

def plot_av(data,ts,std,xlabel = "Temperature (T)",ylabel = 'Phi' ):
    f = plt.figure(figsize=(18, 10))  # plot the calculated values
    ax1 = f.add_subplot(1, 1, 1)
    ax1.scatter(ts, data, s=50, marker='o', color='IndianRed')
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.fill_between(ts,data-std,data+std,color='purple',alpha=0.1)
    ax1.set_xscale('log')
    plt.show()

def subplt_avg(phi,phiSus,mag,sus,ts,path_output,dim,std_phi,std_phi_sus,std_mag,std_sus):#,stdPhi,stdPhiSus,stdmag,stdSus):
    f = plt.figure(figsize=(18, 10))  # plot the calculated values

    ax1 = f.add_subplot(2, 2, 1)
    ax1.scatter(ts, sus, s=50, marker='o', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Susceptibility", fontsize=20)
    plt.fill_between(ts, sus - std_sus, sus + std_sus, color='purple', alpha=0.1)
    ax1.set_xscale('log')

    ax2 = f.add_subplot(2, 2, 2)
    ax2.scatter(ts, phi, s=50, marker='*', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Phi", fontsize=20)
    plt.fill_between(ts, phi - std_phi, phi + std_phi, color='purple', alpha=0.1)
    ax2.set_xscale('log')

    ax3 = f.add_subplot(2, 2, 3)
    ax3.scatter(ts, mag, s=50, marker='*', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Magnetization", fontsize=20)
    plt.fill_between(ts, mag - std_mag, mag + std_mag, color='purple', alpha=0.1)
    ax3.set_xscale('log')

    ax4 = f.add_subplot(2, 2, 4)
    ax4.scatter(ts, phiSus, s=50, marker='*', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Phi_sus", fontsize=20)
    plt.fill_between(ts, phiSus - std_phi_sus, phiSus + std_phi_sus, color='purple', alpha=0.1)
    ax4.set_xscale('log')

    #plt.show()f=
    plt.savefig(path_output + 'plots_'  + str(dim) + '.png', dpi=300)

def avg_phi_stats(dim):

    # Calculate the average phi and summary statistics

    #load the data to be averaged for phi

    base_filepath_ising = '/home/user/Desktop/phiTest/randomTest/'+ str(dim) + '/ising/'
    base_filepath_phi = '/home/user/Desktop/phiTest/randomTest/' + str(dim) + '/phi/'
    path_output = '/home/user/Desktop/phiTest/randomTest/' + str(dim) + '/avg/'
    #Phi files
    filePhi = '/phi.csv'
    filePhiSus = '/phiSus.csv'
    #ising files
    fileMag = '/magn.csv'
    fileSus = '/susc.csv'
    fileTS = '/temps.csv'

    ts = load_matrix(base_filepath_phi + '0' + fileTS)

    dataPhi,dataPhiSus,dataMag,dataSus = [],[],[],[]


    for i in range(19):
        if i == 3:
            i+=1
        if i ==4:
            i+=1
        dataPhi.append(load_matrix(base_filepath_phi + str(i) + filePhi))
        dataPhiSus.append(load_matrix(base_filepath_phi + str(i) + filePhiSus))
        dataMag.append(load_matrix(base_filepath_ising + str(i) + fileMag))
        dataSus.append(load_matrix(base_filepath_ising + str(i) + fileSus))

    data_phi = np.array(dataPhi)
    data_phi_sus = np.array(dataPhiSus)
    data_mag = np.array(dataMag)
    data_sus = np.array(dataSus)

    phiAvg = moving_avg(data_phi)
    phiSusAvg = moving_avg(data_phi_sus)
    magAvg = moving_avg(data_mag)
    susAvg = moving_avg(data_sus)

    phi_std = moving_std(data_phi)
    phi_sus_std = moving_std(data_phi_sus)
    mag_std = moving_std(data_mag)
    sus_std = moving_std(data_sus)

    subplt_avg(phiAvg,phiSusAvg,magAvg,susAvg,ts,path_output,dim,phi_std,phi_sus_std,mag_std,sus_std)

    #plot_av(phiAvg,ts,phi_std)
    #plot_av(phiSusAvg,ts,phi_sus_std,ylabel='Phi Susceptibility')
    #plot_av(magAvg,ts,mag_std,ylabel='Magnetization')
    #plot_av(susAvg,ts,sus_std,ylabel='Magnetic Susceptibility')

avg_phi_stats(3)
avg_phi_stats(4)
avg_phi_stats(5)