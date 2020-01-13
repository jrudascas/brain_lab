import pandas as pd
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg

import matplotlib.pyplot as plt
from projects.phi.tools.utils import *

# The first thing to do is create and store the data as dictionaries to turn into a pandas dataframe
filename = '/home/user/Desktop/phi_csv/anova_csv/everything2.csv'

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

aud_w_baseline = pd.concat([baseline,aud])
cingulo_w_baseline = pd.concat([baseline,cingulo])
dmn_w_baseline = pd.concat([baseline,dmn])
dorsal_w_baseline = pd.concat([baseline,dorsal])
fronto_w_baseline = pd.concat([baseline,fronto])
retro_w_baseline = pd.concat([baseline,retro])
smhand_w_baseline = pd.concat([baseline,smhand])
smmouth_w_baseline = pd.concat([baseline,smmouth])
ventral_w_baseline = pd.concat([baseline,ventral])
vis_w_baseline = pd.concat([baseline,vis])


list_for_anova = [aud_w_baseline,cingulo_w_baseline,dmn_w_baseline,dorsal_w_baseline,fronto_w_baseline,retro_w_baseline,smhand_w_baseline,smmouth_w_baseline,ventral_w_baseline,vis_w_baseline]

summary_aud = rp.summary_cont(aud_w_baseline.groupby('network'))
summary_cingulo = rp.summary_cont(cingulo_w_baseline.groupby('network'))
summary_dmn = rp.summary_cont(dmn_w_baseline.groupby('network'))
summary_dorsal = rp.summary_cont(dorsal_w_baseline.groupby('network'))
summary_fronto = rp.summary_cont(fronto_w_baseline.groupby('network'))
summary_retro = rp.summary_cont(retro_w_baseline.groupby('network'))
summary_smhand = rp.summary_cont(smhand_w_baseline.groupby('network'))
summary_smmouth = rp.summary_cont(smmouth_w_baseline.groupby('network'))
summary_ventral = rp.summary_cont(ventral_w_baseline.groupby('network'))
summary_vis = rp.summary_cont(aud_w_baseline.groupby('network'))



summary_aud2 = rp.summary_cont(aud_w_baseline.groupby(['network','state']))
summary_cingulo2 = rp.summary_cont(cingulo_w_baseline.groupby(['network','state']))
summary_dmn2 = rp.summary_cont(dmn_w_baseline.groupby(['network','state']))
summary_dorsal2 = rp.summary_cont(dorsal_w_baseline.groupby(['network','state']))
summary_fronto2 = rp.summary_cont(fronto_w_baseline.groupby(['network','state']))
summary_retro2 = rp.summary_cont(retro_w_baseline.groupby(['network','state']))
summary_smhand2 = rp.summary_cont(smhand_w_baseline.groupby(['network','state']))
summary_smmouth2 = rp.summary_cont(smmouth_w_baseline.groupby(['network','state']))
summary_ventral2 = rp.summary_cont(ventral_w_baseline.groupby(['network','state']))
summary_vis2 = rp.summary_cont(aud_w_baseline.groupby(['network','state']))

for data in list_for_anova:
    model = ols('phi ~ network*state',data).fit()
    table = sm.stats.anova_lm(model, typ=3)

    print(table)




print(df)