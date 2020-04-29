
from projects.qm_brain.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as ss


stimuli =[('New Data/',15),('BYD/',14),('nyc_data/',13)]

titles = ['Taken Phase','Taken Scrambled Phase','BYD Phase','BYD Scrambled Phase','Rest Phase','Present Phase']

count=0

phase,prob,avg = [],[],[]

cats = ['taken','taken_s','byd','byd_s','rest','present']

for stimulus in stimuli:

    main_path = '/home/user/Desktop/QMBrain/' + stimulus[0]

    x = load_matrix(main_path + 'x_chanloc.csv')
    y = load_matrix(main_path + 'y_chanloc.csv')


    condition_list = ['Cond10/','Cond12/']

    num_sub = stimulus[1]
    num_electrodes = 92

    list_sub = np.arange(num_sub)

    const =  (num_electrodes**2 - num_electrodes)/2

    for condition in condition_list:


        load_path = main_path + condition +'/'

        isc_avg = load_matrix(load_path+'avg_isc.csv')
        isc_avg_prob = load_matrix(load_path+'avg_isc_prob.csv')
        isc_avg_phase = load_matrix(load_path+'avg_isc_phase.csv')

        avg.append(isc_avg)
        prob.append(isc_avg_prob)
        phase.append(isc_avg_phase)

headers_avg = ['taken','taken_s','byd','byd_s','rest','present']
headers_prob = ['taken','taken_s','byd','byd_s','rest','present']
headers_phase = ['taken','taken_s','byd','byd_s','rest','present']

dict_avg = {headers_avg[i]: avg[i] for i in range(len(headers_avg))}
dict_prob = {headers_prob[i]: prob[i] for i in range(len(headers_prob))}
dict_phase = {headers_phase[i]: phase[i] for i in range(len(headers_phase))}

df_avg = pd.DataFrame(dict_avg)
df_prob = pd.DataFrame(dict_prob)
df_phase =pd.DataFrame(dict_phase)


def t_test(a,b,N):
    var_a = np.var(a)
    var_b = np.var(b)

    s = np.sqrt((var_a + var_b) / 2)

    t = (a.mean() - b.mean()) / (s * np.sqrt(2 / N))

    df = 2 * N - 2

    p = 1 - ss.t.cdf(t, df=df)

    return t,p

# Taken my way
t_t, p_t = t_test(df_avg['taken'],df_avg['taken_s'],92)
t_t_prob, p_t_prob = t_test(df_prob['taken'],df_prob['taken_s'],92)
t_t_phase, p_t_phase = t_test(df_phase['taken'],df_phase['taken_s'],92)

# byd my way
t_b, p_b = t_test(df_avg['byd'],df_avg['byd_s'],92)
t_b_prob, p_b_prob = t_test(df_prob['byd'],df_prob['byd_s'],92)
t_b_phase, p_b_phase = t_test(df_phase['byd'],df_phase['byd_s'],92)


#taken s v us
t_taken, p_taken = ss.ttest_ind(df_avg['taken'],df_avg['taken_s'])
t_taken_prob, p_taken_prob = ss.ttest_ind(df_prob['taken'],df_prob['taken_s'])
t_taken_phase, p_taken_phase = ss.ttest_ind(df_phase['taken'],df_phase['taken_s'])

#byd s v us
t_byd, p_byd = ss.ttest_ind(df_avg['byd'],df_avg['byd_s'])
t_byd_prob, p_byd_prob = ss.ttest_ind(df_prob['byd'],df_prob['byd_s'])
t_byd_phase, p_byd_phase = ss.ttest_ind(df_phase['byd'],df_phase['byd_s'])

#byd V RESR
t_byd_rest, p_byd_rest = ss.ttest_ind(df_avg['byd'],df_avg['rest'])
t_byd_rest_prob, p_byd_rest_prob = ss.ttest_ind(df_prob['byd'],df_prob['rest'])
t_byd_rest_phase, p_byd_rest_phase = ss.ttest_ind(df_phase['byd'],df_phase['rest'])

#byd s V RESR
t_byd_s_rest, p_byd_s_rest = ss.ttest_ind(df_avg['byd_s'],df_avg['rest'])
t_byd_s_rest_prob, p_byd_s_rest_prob = ss.ttest_ind(df_prob['byd_s'],df_prob['rest'])
t_byd_s_rest_phase, p_byd_s_rest_phase = ss.ttest_ind(df_phase['byd_s'],df_phase['rest'])


#takem V RESR
t_taken_rest, p_taken_rest = ss.ttest_ind(df_avg['taken'],df_avg['rest'])
t_taken_rest_prob, p_taken_rest_prob = ss.ttest_ind(df_prob['taken'],df_prob['rest'])
t_taken_rest_phase, p_taken_rest_phase = ss.ttest_ind(df_phase['taken'],df_phase['rest'])

#taken s V RESR
t_taken_s_rest, p_taken_s_rest = ss.ttest_ind(df_avg['taken_s'],df_avg['rest'])
t_taken_s_rest_prob, p_taken_s_rest_prob = ss.ttest_ind(df_prob['taken_s'],df_prob['rest'])
t_taken_s_rest_phase, p_taken_s_rest_phase = ss.ttest_ind(df_phase['taken_s'],df_phase['rest'])


# rest v present
t_present_rest, p_present_rest = ss.ttest_ind(df_avg['present'],df_avg['rest'])
t_present_rest_prob, p_present_rest_prob = ss.ttest_ind(df_prob['present'],df_prob['rest'])
t_present_rest_phase, p_present_rest_phase = ss.ttest_ind(df_phase['present'],df_phase['rest'])


# Raw List within stimulus
p_list = [p_taken,p_byd]
t_list = [t_taken,t_byd]

p_list_prob = [p_taken_prob,p_byd_prob]
t_list_prob = [t_taken_prob,t_byd_prob]

p_list_phase = [p_taken_phase,p_byd_phase]
t_list_phase = [t_taken_phase,t_byd_phase]

titl = ['Taken','BYD']

dict_p = {titl[i]: p_list[i] for i in range(len(titl))}
dict_t = {titl[i]: t_list[i] for i in range(len(titl))}

print('Taken',df_avg['taken'].mean())
print('Taken Prob',df_prob['taken'].mean())
print('Taken Phase',df_phase['taken'].mean())

print('Taken S',df_avg['taken_s'].mean())
print('Taken S Prob',df_prob['taken_s'].mean())
print('Taken S Phase',df_phase['taken_s'].mean())

print('BYD',df_avg['byd'].mean())
print('BYD Prob',df_prob['byd'].mean())
print('BYD Phase',df_phase['byd'].mean())

print('BYD S',df_avg['byd_s'].mean())
print('BYD S Prob',df_prob['byd_s'].mean())
print('BYD S Phase',df_phase['byd_s'].mean())



'''
d_p = pd.DataFrame(dict_p)
d_t = pd.DataFrame(dict_t)

print('Raw')
print('P')
print(d_p.to_latex())
print('T')
print(d_t.to_latex())
'''

dict_p_prob = {titl[i]: p_list_prob[i] for i in range(len(titl))}
dict_t_prob = {titl[i]: t_list_prob[i] for i in range(len(titl))}
'''
d_p_prob = pd.DataFrame(dict_p_prob)
d_t_prob = pd.DataFrame(dict_t_prob)

print('Prob')
print('P')
print(d_p_prob.to_latex())
print('T')
print(d_t_prob.to_latex())

'''
dict_p_phase = {titl[i]: p_list_phase[i] for i in range(len(titl))}
dict_t_phase = {titl[i]: t_list_phase[i] for i in range(len(titl))}

d_p_phase = pd.DataFrame(dict_p_phase)
d_t_phase = pd.DataFrame(dict_t_phase)

print('Phase')
print('P')
print(d_p_phase.to_latex())
print('T')
print(d_t_phase.to_latex())
