def to_found_index_tpm(state):
    import numpy as np
    n = len(state)
    setting_int = np.linspace(0, 2**n - 1, num=2**n).astype(int)
    M = list(map(lambda x: list(np.binary_repr(x, width=n)), setting_int))
    M = np.flipud(np.fliplr(np.asarray(M).astype(np.int)))

    return M*2 -1

def load_matrix(file):
    import numpy as np
    import scipy.io

    extension = file.split('.')[-1]
    if str(extension) == 'csv':
        return np.genfromtxt(file,delimiter = ',')
    elif str(extension) == 'npy':
        return np.load(file)
    elif str(extension) == 'mat':
        return scipy.io.loadmat(file)
    elif str(extension) == 'npz':
        return np.load(file)

def makedir2(path):
    import os
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return True

def file_exists(filename):
    import os
    exists = os.path.isfile(filename)
    if exists:
        return True
    else:
        return False

def save_ts(ts, path_output, filepath, sub_num):
    import numpy as np
    if makedir2(path_output):
        default_delimiter = ','
        format = '%1.5f'
        name = filepath.split('.')[-3]
        filename = path_output + '/' + 'ts_' + name + '_Sub' + str(sub_num) + '.csv'
        if not file_exists(filename):
            np.savetxt(filename, ts, delimiter=default_delimiter, fmt=format)

def save_Jij(Jij, save_path, sub_num):
    import numpy as np
    if makedir2(save_path):
        default_delimiter = ','
        format = '%1.5f'

        filename = save_path + '/' + 'Jij' + '_Sub' + str(sub_num) + '.csv'
        if not file_exists(filename):
            np.savetxt(filename, Jij, delimiter=default_delimiter, fmt=format)

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

def save_freq(frequency,path_output,num):
    import numpy as np
    import matplotlib.pyplot as plt
    path_output = path_output + '/data'
    if makedir2(path_output):
        default_delimiter = ','
        format = '%1.5f'

        filename1 =path_output + '/' + 'freq_'+ str(num) + '.csv'

        if not file_exists(filename1):
            np.savetxt(filename1, frequency, delimiter=default_delimiter, fmt=format)

        plt.bar(np.arange(len(frequency)),frequency)

        filename2 = path_output + '/' + 'plots_hist_' + str(num)+'.png'
        if not file_exists(filename2):
            plt.savefig(filename2, dpi=1200)

        plt.close()

def tpm_SbyN(time_series):
    import numpy as np
    avgs = np.mean(time_series, axis=0)

    time_series_other = np.copy(time_series)
    for i in range(len(avgs)):
        time_series[np.where(time_series_other[:, i] >= avgs[i]),i] = 0
        time_series[np.where(time_series_other[:, i] < avgs[i]),i] = 1

    time_series = time_series.astype(np.int)

    markov_chain = time_series.tolist()
    n = len(markov_chain[0])
    tpm = np.zeros((2 ** n, n))

    for (s1, s2) in zip(markov_chain, markov_chain[1:]):
        i = int(''.join(map(str, s1)), 2)
        for k in range(len(s1)):
            if s1[k] != s2[k]:
                tpm[i][k] += 1

    state_total = np.sum(tpm, axis=-1)


    frequency = np.zeros((2 ** time_series.shape[-1]))

    for s in markov_chain:
        i = int(''.join(map(str, s)), 2)
        frequency[i] += 1

    frequency /= len(markov_chain)



    return np.copy(tpm), np.copy(state_total), np.copy(frequency)

def tpm_SbyN_2(time_series):
    import numpy as np
    avgs = np.mean(time_series, axis=0)

    time_series_other = np.copy(time_series)
    for i in range(len(avgs)):
        time_series[np.where(time_series_other[:, i] >= avgs[i]),i] = 0
        time_series[np.where(time_series_other[:, i] < avgs[i]),i] = 1

    time_series = time_series.astype(np.int)

    markov_chain = time_series.tolist()
    n = len(markov_chain[0])
    tpm = np.zeros((2 ** n, n))

    norm = np.zeros((1,2**n))

    for (s1, s2) in zip(markov_chain, markov_chain[1:]):
        i = int(''.join(map(str, s1)), 2)
        norm[i]+=1
        for k in range(len(s1)):
            if s1[k] != s2[k]:
                tpm[i][k] += 1

    state_total = np.sum(tpm, axis=-1)


    frequency = np.zeros((2 ** time_series.shape[-1]))

    for s in markov_chain:
        i = int(''.join(map(str, s)), 2)
        frequency[i] += 1

    frequency /= len(markov_chain)

    for ind,normal in enumerate(norm):
        tpm[ind,...] = tpm[ind,...]/normal



    return np.copy(tpm), np.copy(state_total), np.copy(frequency)


def main_tpm_branch(time_series):
    import numpy as np
    avgs = np.mean(time_series, axis=0)

    time_series_other = np.copy(time_series)
    for i in range(len(avgs)):
        time_series[np.where(time_series_other[:, i] >= avgs[i]),i] = 0
        time_series[np.where(time_series_other[:, i] < avgs[i]),i] = 1

    time_series = time_series.astype(np.int)

    markov_chain = time_series.tolist()
    n = len(markov_chain[0])
    tpm = np.zeros((2 ** n, 2**n))

    for (s1, s2) in zip(markov_chain, markov_chain[1:]):
        i = int(''.join(map(str, s1)), 2)
        j = int(''.join(map(str, s2)), 2)
        tpm[i][j] += 1

    state_total = np.sum(tpm, axis=-1)


    frequency = np.zeros((2 ** time_series.shape[-1]))

    for s in markov_chain:
        i = int(''.join(map(str, s)), 2)
        frequency[i] += 1

    frequency /= len(markov_chain)

    return np.copy(tpm), np.copy(state_total), np.copy(frequency)

def plot_ts_avg(ts,path_output = None,num=None):
    import numpy as np
    import matplotlib.pyplot as plt

    y = np.mean(ts, axis=0)
    x = np.arange(ts.shape[0])

    f = plt.figure()
    plt.subplot(ts.shape[-1],1,1)
    plt.plot(x,ts[:,0])
    plt.axhline(y[0],color='r')

    plt.subplot(ts.shape[-1],1,2)
    plt.plot(x,ts[:,1])
    plt.axhline(y[1],color='r')

    plt.subplot(ts.shape[-1],1,3)
    plt.plot(x,ts[:,2])
    plt.axhline(y[2],color='r')

    plt.subplot(ts.shape[-1],1,4)
    plt.plot(x,ts[:,3])
    plt.axhline(y[3],color='r')

    plt.subplot(ts.shape[-1],1,5)
    plt.plot(x,ts[:,4])
    plt.axhline(y[4],color='r')

    if path_output is not None and num is not None:
        if makedir2(path_output):
            path_output = path_output + '/data'

            filename = path_output + '/' + 'plots_avg_ts_' + str(num)+'.png'
            if not file_exists(filename):
                plt.savefig(filename, dpi=1200)

            plt.close()

def empirical_tpm_og(time_series,path_output,tpm_count):
    import numpy as np
    from nilearn import plotting
    tpm, state_total,frequency = main_tpm_branch(time_series)

    # Normalizing respect to rows
    for div in range(len(state_total)):
        if state_total[div] != 0.0:
            tpm[div, :] /= state_total[div]

    save_tpm(tpm, path_output, tpm_count)
    save_freq(frequency, path_output, tpm_count)
    plot_ts_avg(time_series, path_output=path_output, num=tpm_count)

    return tpm

def empirical_tpm_eps(time_series,path_output,tpm_count):
    import numpy as np

    tpm, state_total,frequency = main_tpm_branch(time_series)

    eps = np.min(time_series[np.isfinite(np.log10(time_series))]) * 0.1
    rando = np.random.rand(1, tpm.size - np.count_nonzero(tpm))
    for rand in rando:
        tpm[np.where(tpm == 0)] = eps * rand

    # Normalizing respect to rows
    for div in range(len(state_total)):
        if state_total[div] != 0.0:
            tpm[div, :] /= state_total[div]


    save_tpm(tpm, path_output, tpm_count)
    save_freq(frequency, path_output, tpm_count)
    plot_ts_avg(time_series, path_output=path_output, num=tpm_count)

    return tpm

def empirical_tpm_concat(time_series,path_output):
    import numpy as np

    assert len(time_series.shape) == 3



    tpm_list = []

    tpm_count = 0

    for i in range(time_series.shape[-1]):

        new_ts = np.delete(time_series,i,axis = 2)

        big_ts_array = np.squeeze(np.vstack(np.array((np.dsplit(new_ts,new_ts.shape[-1])))))

        tpm, state_total, frequency = tpm_SbyN(np.copy(big_ts_array))

        # Normalizing respect to rows
        for div in range(len(state_total)):
            if state_total[div] != 0.0:
                tpm[div, :] /= state_total[div]

        tpm_list.append(tpm)
        tpm_count+=1
        save_tpm(tpm,path_output,tpm_count)
        save_freq(frequency,path_output,tpm_count)
        plot_ts_avg(big_ts_array,path_output=path_output,num=tpm_count)


    return tpm_list

def avg_Jij(Jij):
    import numpy as np
    from projects.generalize_ising_model.tools.utils import to_normalize

    assert len(Jij.shape) == 3

    avg_Jij = np.mean(Jij,axis=-1)

    J = to_normalize(avg_Jij)

    return J

def make_ts_array(path,number_regions = 5):
    import os
    import numpy as np
    tup = ()
    for file in os.listdir(path):
        if file.endswith('.csv'):
            filepath = path + '/' + file
            ts = load_matrix(filepath)
            timeSeries = ts[:, 0:number_regions].astype(np.float32)
            tup = tup + (timeSeries,)

    return np.dstack(tup)

def make_Jij_array(path,number_regions = 5):
    import os
    import numpy as np
    tup = ()
    for file in os.listdir(path):
        if file.endswith('.csv'):
            filepath = path + '/' + file
            Jij = load_matrix(filepath)
            tup = tup + (Jij,)

    return np.dstack(tup)

def to_calculate_mean_phi(tpm, spin_mean,eps=None):
    import numpy as np
    import pyphi
    from pyphi.compute import phi

    rows, columns = tpm.shape

    setting_int = np.linspace(0, rows - 1, num=rows).astype(int)

    M = list(map(lambda x: list(np.binary_repr(x, width=columns)), setting_int))
    #M = np.flipud(np.fliplr(np.asarray(M).astype(np.int)))
    M = np.asarray(M).astype(np.int)

    #num_states = np.log2(N)
    phi_values = []

    network = pyphi.Network(tpm)
    for state in range(rows):
        if eps == None:
            if spin_mean[state] != 0:
                phi_values.append(phi(pyphi.Subsystem(network, M[state, :], range(network.size))))
        else:
            if spin_mean[state] < eps:
                phi_values.append(phi(pyphi.Subsystem(network, M[state, :], range(network.size))))

    weigth = spin_mean[np.where(spin_mean != 0)]

    phiSum = np.sum(phi_values*weigth)

    return np.mean(phi_values), phiSum

def to_save_phi(phi , phiSum, num, path_output):
    import numpy as np

    default_delimiter = ','
    format = '%1.5f'

    filePhi = path_output + 'phi_' + str(num) + '.csv'
    filePhiSum = path_output + 'phiSum_' + str(num) + '.csv'

    if makedir2(path_output):

        if not file_exists(filePhi):
            np.savetxt(filePhi, np.asarray([phi]), delimiter=default_delimiter, fmt=format)
        if not file_exists(filePhiSum):
            np.savetxt(filePhiSum, np.asarray([phiSum]), delimiter=default_delimiter, fmt=format)

def save_list(list,path_output,state,network,type='tpm'):
    import numpy as np

    path_output = path_output + str(network) + '/'

    if makedir2(path_output):
        default_delimiter = ','
        format = '%1.5f'
        filename1 = path_output + '/' + state + type

        if not file_exists(filename1):
            if type == 'freq':
                np.savetxt(filename1 + '.csv', np.asarray(list), delimiter=default_delimiter, fmt=format)
            elif type == 'tpm':
                np.save(filename1,np.asarray(list))

def viola_plot_n_save(list,path,title):
    import numpy as np
    import matplotlib.pyplot as plt

    filename1 = path + '/' + title + '.npy'
    if makedir2(path):

        if not file_exists(filename1):
            np.save(filename1, np.asarray([list]))

    tick_pos = np.arange(len(list))
    tick_title = ['Awake','Mild','Deep','Recovery']
    fig = plt.figure()
    plt.violinplot(list, tick_pos,showmeans=True, showextrema=True, showmedians=False)
    plt.title(title +' of TPM')
    plt.xticks(tick_pos,tick_title)

    filename2 = path + '/' + title + '.png'

    if not file_exists(filename2):
        plt.savefig(filename2)

    plt.close(fig)

def save_hubs(hub_list,path):
    import numpy as np

    filename1 = path + '/Hubs'

    if makedir2(path):
        if not file_exists(filename1+'.npy'):
            np.save(filename1,np.asarray([hub_list]))

def to_find_critical_temperature(data, temp, fit_type='rel_extrema'):
    import numpy as np
    from scipy.signal import argrelextrema

    y_test = data.copy()

    if fit_type == 'rel_extrema':
        local_max = argrelextrema(data, np.greater, order=1)

        y_test[local_max] = (y_test[np.array(local_max) + 1] + y_test[np.array(local_max) - 1]) / 2
        return temp[np.where(y_test == np.max(y_test[local_max]))]

    elif fit_type == 'max':
        local_max = np.where(data == max(data))
        return data[np.where(y_test == np.max(y_test[local_max]))]