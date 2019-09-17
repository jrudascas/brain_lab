from projects.phi.tools.utils import load_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator

main_path = '/home/user/Desktop/Popiel/check_ising/'

sizes = [5,10,25,100,250]

temp_params = [(-3,4,50),(-1,8,50),(0,12,50), (1,100,50),(1.3,200,50)]

colourWheel =['#329932',
            '#ff6961',
            'b',
            '#6a3d9a',
            '#fb9a99',
            '#e31a1c',
            '#fdbf6f',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928',
            '#67001f',
            '#b2182b',
            '#d6604d',
            '#f4a582',
            '#fddbc7',
            '#f7f7f7',
            '#d1e5f0',
            '#92c5de',
            '#4393c3',
            '#2166ac',
            '#053061']
dashesStyles = [[3,1],
            [1000,1],
            [2,1,10,1],
            [4, 1, 1, 1, 1, 1]]


dict = {}

for ind,size in enumerate(sizes):
    dict['susc'] = load_matrix(main_path+str(size)+'/susc.csv')
    dict['heat'] = load_matrix(main_path + str(size) + '/heat.csv')

    x = np.logspace(temp_params[ind][0],np.log10(temp_params[ind][1]),num=temp_params[ind][2])

    df = pd.DataFrame(dict)

    plt.close('all')
    fig, ax = plt.subplots()
    for j, series in enumerate(df.columns):
        if (series == 'susc'):
            alphaVal = 1.
            linethick = 5
        else:
            alphaVal = 0.6
            linethick = 3.5
        ax.plot(x,
                df[series],
                color=colourWheel[j % len(colourWheel)],
                linestyle='-',
                dashes=dashesStyles[j % len(dashesStyles)],
                lw=linethick,
                label=series,
                alpha=alphaVal)
    ax.set_xlabel('')
    #ax.set_xscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.major.formatter._useMathText = True
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_label_coords(0.63, 1.01)
    ax.yaxis.tick_right()
    #nameOfPlot = 'Criticality and the Ising model (' + str(size) + 'by' + str(size) + 'random matrix)'
    #plt.ylabel(nameOfPlot, rotation=0)
    ax.legend(frameon=False, loc='upper left', ncol=2, handlelength=4)
    plt.savefig(main_path+str(size)+'plot.pdf', dpi=300)
    plt.show()


