from pyAudioAnalysis import audioBasicIO
from dcor import distance_correlation, u_distance_correlation_sqr
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from nilearn.signal import clean
from neuroimaging_pipelines.tools.utils import dspmt

audio_path = "/home/brainlab/Desktop/Rudas/Data/Propofol/Taken-[AudioTrimmer.com].wav"
time_series_path = '/home/brainlab/Desktop/Rudas/Data/Propofol/Awake/Task/output/datasink/preprocessing/sub-2014_05_02_02CB/time_series.csv'
predictors_audio = '/home/brainlab/mritoolbox_predictors.csv'

[Fs, raw_audio] = audioBasicIO.readAudioFile(audio_path)
predictors = np.loadtxt(predictors_audio, delimiter=',')
task = np.loadtxt(time_series_path, delimiter=',')

ts_index = [28,77]

#hrf = dspmt(np.linspace(0,5, num=4))

#predictors_hrf = np.zeros(predictors.shape)
#for index in range(predictors.shape[1]):
#    predictors_hrf[:, index] = np.convolve(predictors[:, index], hrf)[:150]

predictor_cleaned = clean(signals=predictors,
                          detrend=True,
                          standardize=True,
                          low_pass=0.1,
                          high_pass=0.01,
                          t_r=2,
                          ensure_finite=True)

np.savetxt('/home/brainlab/mritoolbox_predictors_normalized.csv', predictor_cleaned, fmt='%10.6f', delimiter=',')

predictor_cleaned_hrf = clean(signals=downsample_predictor_hrf,
                          detrend=True,
                          standardize=True,
                          low_pass=0.1,
                          high_pass=0.01,
                          t_r=2,
                          ensure_finite=True)

print(pearsonr(task[:, 44], predictor_cleaned_hrf[:, 5])[0])
plt.plot(task[:, 44], c='orange')
plt.plot(predictor_cleaned_hrf[:, 7], c='red')

plt.show()


#dc_list = []
#for roi in range(task.shape[1]):
#    dc_list.append((roi, u_distance_correlation_sqr(task[:, roi], predictor_cleaned)))
#    print(str(roi) + ': ' + str(u_distance_correlation_sqr(task[:, roi], predictor_cleaned)))

#dc_list.sort(key=lambda roi: roi[1])
#print(dc_list[-10:-1])


for index in range(predictors.shape[1]):
    for roi in range(task.shape[1]):
        if pearsonr(task[:, roi], predictor_cleaned_hrf[:, index])[0] > 0.4:
            print('Predictor: ' + str(index) + ' -- ROI:' + str(roi))
            print(pearsonr(task[:, roi], predictor_cleaned_hrf[:, index]))