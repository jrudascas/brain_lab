import pandas as pd
import numpy as np
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns

import matplotlib.pyplot as plt
from projects.phi.tools.utils import *

# The first thing to do is create and store the data as dictionaries to turn into a pandas dataframe
filename = '/home/user/Desktop/phi_csv/anova_csv/everything.csv'

df = pd.read_csv(filename)

#print(df)

aud = df.loc[df.network == 'aud']
baseline = df.loc[df.network == 'baseline']
cingulo = df.loc[df.network == 'cingulo']
dmn = df.loc[df.network == 'dmn']
dorsal = df.loc[df.network == 'dorsal']
fronto = df.loc[df.network == 'fronto']
retro = df.loc[df.network == 'retro']
smhand = df.loc[df.network == 'smhand']
smmouth = df.loc[df.network == 'smmouth']
ventral = df.loc[df.network == 'ventral']
vis = df.loc[df.network == 'vis']

aud_matrix = np.array([aud['awake'],aud['mild'],aud['deep'],aud['recovery']]).T
baseline_matrix = np.array([baseline['awake'],baseline['mild'],baseline['deep'],baseline['recovery']]).T
cingulo_matrix = np.array([cingulo['awake'],cingulo['mild'],cingulo['deep'],cingulo['recovery']]).T
dmn_matrix = np.array([dmn['awake'],dmn['mild'],dmn['deep'],dmn['recovery']]).T
dorsal_matrix = np.array([dorsal['awake'],dorsal['mild'],dorsal['deep'],dorsal['recovery']]).T
fronto_matrix = np.array([fronto['awake'],fronto['mild'],fronto['deep'],fronto['recovery']]).T
retro_matrix = np.array([retro['awake'],retro['mild'],retro['deep'],retro['recovery']]).T
smhand_matrix = np.array([smhand['awake'],smhand['mild'],smhand['deep'],smhand['recovery']]).T
smmouth_matrix = np.array([smmouth['awake'],smmouth['mild'],smmouth['deep'],smmouth['recovery']]).T
ventral_matrix = np.array([ventral['awake'],ventral['mild'],ventral['deep'],ventral['recovery']]).T
vis_matrix = np.array([vis['awake'],vis['mild'],vis['deep'],vis['recovery']]).T

'''
plt.close()
plt.violinplot(baseline_matrix)
plt.show()
'''

test_vector = np.array([2,1,-1,1])

len_test_vec = np.linalg.norm(test_vector)

phi_dist_aud = aud_matrix@test_vector
phi_dist_baseline = baseline_matrix@test_vector
phi_dist_cingulo = cingulo_matrix@test_vector
phi_dist_dmn = dmn_matrix@test_vector
phi_dist_dorsal = dorsal_matrix@test_vector
phi_dist_fronto = fronto_matrix@test_vector
phi_dist_retro = retro_matrix@test_vector
phi_dist_smhand = smhand_matrix@test_vector
phi_dist_smmouth = smmouth_matrix@test_vector
phi_dist_ventral = ventral_matrix@test_vector
phi_dist_vis = vis_matrix@test_vector

aud_norm = np.linalg.norm(aud_matrix,axis=1)
cingulo_norm = np.linalg.norm(cingulo_matrix,axis=1)
baseline_norm = np.linalg.norm(baseline_matrix,axis=1)
dmn_norm = np.linalg.norm(dmn_matrix,axis=1)
dorsal_norm = np.linalg.norm(dorsal_matrix,axis=1)
fronto_norm = np.linalg.norm(fronto_matrix,axis=1)
retro_norm = np.linalg.norm(retro_matrix,axis=1)
smhand_norm = np.linalg.norm(smhand_matrix,axis=1)
smmouth_norm = np.linalg.norm(smmouth_matrix,axis=1)
ventral_norm = np.linalg.norm(ventral_matrix,axis=1)
vis_norm = np.linalg.norm(vis_matrix,axis=1)

cos_dist_aud = phi_dist_aud/(aud_norm*len_test_vec)
cos_dist_baseline = phi_dist_baseline/(baseline_norm*len_test_vec)
cos_dist_cingulo = phi_dist_cingulo/(cingulo_norm*len_test_vec)
cos_dist_dmn = phi_dist_dmn/(dmn_norm*len_test_vec)
cos_dist_dorsal = phi_dist_dorsal/(dorsal_norm*len_test_vec)
cos_dist_fronto = phi_dist_fronto/(fronto_norm*len_test_vec)
cos_dist_retro = phi_dist_retro/(retro_norm*len_test_vec)
cos_dist_smhand = phi_dist_smhand/(smhand_norm*len_test_vec)
cos_dist_smmouth = phi_dist_smmouth/(smmouth_norm*len_test_vec)
cos_dist_ventral = phi_dist_ventral/(ventral_norm*len_test_vec)
cos_dist_vis = phi_dist_vis/(vis_norm*len_test_vec)

sns.distplot(cos_dist_aud)
plt.title('Cosine Distribution Auditory Network')
plt.show()
sns.distplot(cos_dist_baseline)
plt.title('Cosine Distribution Random Sample')
plt.show()
sns.distplot(cos_dist_cingulo)
plt.title('Cosine Distribution Cingulate Network')
plt.show()
sns.distplot(cos_dist_dmn)
plt.title('Cosine Distribution DMN')
plt.show()
sns.distplot(cos_dist_dorsal)
plt.title('Cosine Distribution Dorsal Network')
plt.show()
sns.distplot(cos_dist_fronto)
plt.title('Cosine Distribution Frontoparietal Network')
plt.show()
sns.distplot(cos_dist_retro)
plt.title('Cosine Distribution Retrosplenial Network')
plt.show()
sns.distplot(cos_dist_smhand)
plt.title('Cosine Distribution SMHand Network')
plt.show()
sns.distplot(cos_dist_smmouth)
plt.title('Cosine Distribution SMMouth Network')
plt.show()
sns.distplot(cos_dist_ventral)
plt.title('Cosine Distribution Ventral Network')
plt.show()
sns.distplot(cos_dist_vis)
plt.title('Cosine Distribution Visual Network')
plt.show()