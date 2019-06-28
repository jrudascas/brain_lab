from projects.qm_brain.utils import  *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf',FigureCanvasPgf)


times= load_matrix('/home/user/Desktop/QMBrain/New Data/times.csv')

uncertainity_x1 = load_matrix('/home/user/Desktop/QMBrain/New Data/Cond10/10/DeltaXDeltaPX.csv')
uncertainity_y1 = load_matrix('/home/user/Desktop/QMBrain/New Data/Cond10/10/DeltaYDeltaPY.csv')



minimum_x = np.min(uncertainity_x1)
minimum_y = np.min(uncertainity_y1)

print('The minimum uncertainty for x is: ', minimum_x)
print('The minimum uncertainty for y is: ', minimum_y)

print('The maximum uncertainty for x is: ', np.max(uncertainity_x1))
print('The maximum uncertainty for y is: ', np.max(uncertainity_y1))

plt.plot()


pgf_with_latex = {
    "pgf.texsystem": "xelatex",         # use Xelatex which is TTF font aware
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",             # use serif rather than sans-serif
    "font.serif": "Ubuntu",             # use 'Ubuntu' as the standard font
    "font.sans-serif": [],
    "font.monospace": "Ubuntu Mono",    # use Ubuntu mono if we have mono
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
    "text.latex.unicode": True,
    "pgf.preamble": [
        r'\usepackage{fontspec}',
        r'\setmainfont{Ubuntu}',
        r'\setmonofont{Ubuntu Mono}',
        r'\usepackage{unicode-math}',
        r'\setmathfont{Ubuntu}'

    ]
}

matplotlib.rcParams.update(pgf_with_latex)

'''
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
'''
plt.clf()
plt.close()
fig = plt.figure()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.9)

fig.add_subplot(2,2,1)
plt.plot(times, uncertainity_x1)
plt.ylim([0,4])

fig.add_subplot(2,2,2)
plt.plot(times, uncertainity_x2)
plt.ylim([0,4])

fig.add_subplot(2,2,3)
plt.plot(times, uncertainity_x3)
plt.ylim([0,4])

fig.add_subplot(2,2,4)
plt.plot(times, uncertainity_x4)
plt.ylim([0,4])


#ax.set_ylabel(r'$\Delta x(t) \Delta p_x(t)$')
#ax.set_xlabel(r'Time $(\mu s)$')

fig.text(0.5, 0.04, r'Time $(\mu s)$', ha='center')
fig.text(0.04, 0.5, r'$\Delta x(t) \Delta p_x(t)$', va='center', rotation='vertical')


plt.suptitle(r'Uncertainty Relation in EEG Data (x)')


fig.savefig('uncertaintyXTrunc.pdf')

'''
fig = plt.figure()
fig.add_subplot(2,2,1)
plt.plot(times,uncertainity_x)
plt.ylim([0,2])
plt.show()

plt.plot(times,uncertainity_y)
#plt.ylim([0,2])
plt.show()

'''