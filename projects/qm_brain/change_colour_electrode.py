from projects.qm_brain.utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def get_your_animation(xIn,yIn,max_inds,path):

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(-10, 10), ylim=(-8, 8))
    line, = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    xIn = xIn
    yIn = yIn

    def init():
        line.set_data(xIn,yIn)
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        x = xIn[max_inds[i]]
        y = yIn[max_inds[i]]
        line.set_data(x, y)
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=200, interval=20, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save(path+'plot.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

    plt.show()

num_subjects = 15
main_path = '/home/user/Desktop/QMBrain/New Data/'

filepathX = main_path + 'x_chanloc.csv'
filepathY = main_path + 'y_chanloc.csv'
filepathTimes = main_path + 'times.csv'

times = load_matrix(filepathTimes)
x = load_matrix(filepathX)
y = load_matrix(filepathY)


condition_list = ['Cond10/','Cond12/']

'''

for condition in condition_list:

    for i in range(num_subjects):

        subject_path = main_path + condition + str(i + 1) + '/'

        sub_num = i+1

        cond = condition.split('/')[0]

        print('Running for subject ', i + 1, 'in folder ', condition)

        filepathData = subject_path + 'data.csv'

        data = load_matrix(filepathData)

        phase, normAmp, probability = process_eeg_data(data)

        max_inds = np.argmax(probability, axis=1)

        animate_anigreat(max_inds,x,y,sub_num,cond)

        break
    break

'''

subject_path = main_path + 'Cond10/1/'

sub_num = 1

cond = 'Cond10'

filepathData = subject_path + 'data.csv'

data = load_matrix(filepathData)

phase, normAmp, probability = process_eeg_data(data)

max_inds = np.argmax(probability, axis=1)

animate_anigreat(max_inds, x, y, sub_num, cond)
