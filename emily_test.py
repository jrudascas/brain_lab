import statsmodels.api as sa
import statsmodels.formula.api as sfa
import scikit_posthocs as sp
import numpy as np
import pandas as pd

df = sa.datasets.get_rdataset('iris').data

print(df.head())

ttest = sp.posthoc_ttest(df,val_col='Sepal.Width', group_col='Species', p_adjust='holm')

tukey = sp.posthoc_tukey_hsd()

