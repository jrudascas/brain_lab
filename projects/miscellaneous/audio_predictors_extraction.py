from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

#audio_path = "/home/brainlab/Desktop/Rudas/Data/Propofol/Taken-[AudioTrimmer.com].wav"
audio_path =  "/home/brainlab/Desktop/Rudas/Data/Propofol/Taken-[AudioTrimmer.com].wav"


[Fs, x] = audioBasicIO.readAudioFile(audio_path)
x = audioBasicIO.stereo2mono(x)

tr = 2

F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, tr*Fs, tr*Fs)

np.savetxt('audio_predictors.txt', np.transpose(F[:21]), fmt='%10.6f', delimiter=',')

from nilearn.signal import clean
#F = clean(signals=F,
#          detrend=False,
#          standardize=True,
#          ensure_finite=False)

#for feature in range(2):
#    plt.subplot(2,1,feature+1);
#    plt.plot(F[feature,:]);
    #plt.xlabel('Frame no');
    #plt.ylabel(f_names[feature])
#plt.show()



n = 150
average_signal = []
average_signal_ = []
signal_portion_average = []
signal_portion_mode = []

for i in range(n):
    start = Fs*(2*i)
    end = Fs*(2*i + 2)

    sub_signal = x[start:end]
    sub_signal_ = x[start:int(start + Fs)]


    average_signal.append(np.mean(np.abs(sub_signal)))
    average_signal_.append(np.mean(sub_signal))
    #signal_portion_average.append(np.mean(np.abs(sub_signal_)))
    #average_signal_mod.append(np.median(np.abs(sub_signal_)))
    #signal_portion_mode.append(stats.mode(np.abs(sub_signal_)))

np.savetxt('audio_average_abs.txt', np.transpose(average_signal), fmt='%10.6f', delimiter=' ')
np.savetxt('audio_average.txt', np.transpose(average_signal_), fmt='%10.6f', delimiter=' ')
#np.savetxt('audio_mode_complete_tr.txt', np.transpose(signal_portion_mode), fmt='%10.6f', delimiter=' ')

