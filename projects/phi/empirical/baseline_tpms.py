from projects.phi.tools.utils import *
import os
import numpy as np

brain_states = ['Awake','Deep','Mild','Recovery']
                # ['Recovery']

num_perm = 100
num_nodes = 5

for state in brain_states:
    path_output = '/home/user/Desktop/data_phi/Propofol/Baseline/' + state
    ts_path = path_output + '/ts_all.csv'
    big_ts = load_matrix(ts_path)
    permutation_ts_tpm(big_ts,path_output,num_perm=num_perm,num_nodes=num_nodes)
