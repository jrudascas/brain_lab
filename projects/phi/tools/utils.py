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

def save_ts(ts, path_output, filepath, sub_num):
    import numpy as np
    if makedir2(path_output):
        default_delimiter = ','
        format = '%1.5f'
        name = filepath.split('.')[-3]
        np.savetxt(path_output + '/' + 'ts_' + name + '_Sub' + str(sub_num) + '.csv', ts, delimiter=default_delimiter, fmt=format)

def save_tpm(tpm,path_output,num):
    import numpy as np
    from nilearn import plotting
    import matplotlib.pyplot as plt
    path_output = path_output + '/data'
    if makedir2(path_output):
        default_delimiter = ','
        format = '%1.5f'
        np.savetxt(path_output + '/' + 'tpm_'+ str(num) + '.csv', tpm, delimiter=default_delimiter, fmt=format)

        '''
        plt.imshow(tpm,cmap='plasma')
        plt.colorbar(boundaries=np.linspace(0, 1, 6))
        plt.savefig(path_output + '/' + 'plots_' + str(num)+'.png', dpi=1200)

        plt.close()
        '''
        fig, ax = plt.subplots()

        plotting.plot_matrix(tpm, figure=fig, vmax=1, vmin=0)

        fig.savefig(path_output + '/' + 'plots_' + str(num)+'.png', dpi=1200)

        plt.close(fig)

def save_freq(frequency,path_output,num):
    import numpy as np
    import matplotlib.pyplot as plt
    path_output = path_output + '/data'
    if makedir2(path_output):
        default_delimiter = ','
        format = '%1.5f'
        np.savetxt(path_output + '/' + 'freq_'+ str(num) + '.csv', frequency, delimiter=default_delimiter, fmt=format)

        plt.hist(frequency.T)
        plt.savefig(path_output + '/' + 'plots_hist_' + str(num)+'.png', dpi=1200)

        plt.close()

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
    tpm = np.zeros((2 ** n, 2 ** n))

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

def plot_ts_avg(ts):
    import numpy as np
    import matplotlib.pyplot as plt

    y = np.mean(ts, axis=0)
    x = np.arange(ts.shape[0])

    f = plt.figure()
    plt.subplot(1,ts.shape[-1],1)
    plt.plot(x,ts[:,0])
    plt.axhline(y[0],color='r')

    plt.subplot(1,ts.shape[-1],2)
    plt.plot(x,ts[:,1])
    plt.axhline(y[1],color='r')

    plt.subplot(1,ts.shape[-1],3)
    plt.plot(x,ts[:,2])
    plt.axhline(y[2],color='r')

    plt.subplot(1,ts.shape[-1],4)
    plt.plot(x,ts[:,3])
    plt.axhline(y[3],color='r')

    plt.subplot(1,ts.shape[-1],5)
    plt.plot(x,ts[:,4])
    plt.axhline(y[4],color='r')

    plt.show()

def empirical_tpm_og(time_series):
    import numpy as np
    from nilearn import plotting
    tpm, state_total = main_tpm_branch(time_series)

    # Normalizing respect to rows
    for div in range(len(state_total)):
        if state_total[div] != 0.0:
            tpm[div, :] /= state_total[div]

    plotting.plot_matrix(tpm, colorbar=True, cmap='plasma')
    plotting.show()
    print(tpm)
    return tpm

def empirical_tpm_eps(time_series):
    import numpy as np
    from nilearn import plotting
    tpm, state_total = main_tpm_branch(time_series)

    eps = np.min(time_series[np.isfinite(np.log10(time_series))]) * 0.1
    rando = np.random.rand(1, tpm.size - np.count_nonzero(tpm))
    for rand in rando:
        tpm[np.where(tpm == 0)] = eps * rand

    # Normalizing respect to rows
    for div in range(len(state_total)):
        if state_total[div] != 0.0:
            tpm[div, :] /= state_total[div]

    plotting.plot_matrix(tpm, colorbar=True, cmap='plasma')
    plotting.show()
    print(tpm)

    return tpm

def empirical_tpm_concat(time_series,path_output):
    import numpy as np

    assert len(time_series.shape) == 3


    tpm_list = []

    tpm_count = 0

    for i in range(time_series.shape[-1]):

        new_ts = np.delete(time_series,i,axis = 2)

        big_ts_array = np.squeeze(np.vstack(np.array((np.dsplit(new_ts,new_ts.shape[-1])))))

        tpm, state_total, frequency = main_tpm_branch(np.copy(big_ts_array))

        # Normalizing respect to rows
        for div in range(len(state_total)):
            if state_total[div] != 0.0:
                tpm[div, :] /= state_total[div]

        tpm_list.append(tpm)
        tpm_count+=1
        save_tpm(tpm,path_output,tpm_count)
        save_freq(frequency,path_output,tpm_count)
        #plot_ts_avg(big_ts_array)


    return tpm_list

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

