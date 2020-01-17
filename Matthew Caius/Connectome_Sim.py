from projects.generalize_ising_model.core import generalized_ising
from projects.generalize_ising_model.tools.utils import *
from projects.phi.tools.utils import *
import matplotlib as plt
import numpy as np
import time

#where data comes from
input_path = '/home/brainlab/Desktop/Matt/ConnectomeData/'
#where it will go
output_path = '/home/brainlab/Desktop/Matt/ConnectomeOutput/Test_1/'

#temporary stuff
file_name = "collab_data.csv"
Output_filename = "CollabJij_"

#fetch the Jij csv file
J = to_normalize(np.loadtxt(input_path + file_name))


# Ising Parameters
temperature_parameters = (0.002, 3, 50)  # Temperature parameters (initial tempeture, final temperature, number of steps)
no_simulations = 1200  # Number of simulations after thermalization
thermalize_time = 0.3  #

start_time = time.time()

Simulated_FC, Critical_Temperature, E, M, S, H, Mean_Spin, time_course = generalized_ising(J,
                                                                   temperature_parameters = temperature_parameters,
                                                                   n_time_points= no_simulations,
                                                                   thermalize_time = thermalize_time,
                                                                   phi_variables = True,
                                                                   return_tc= True,
                                                                   type = "digital")

final_time = time.time()
delta_time = final_time - start_time

print("It took " + str(delta_time) + " seconds to calculate the simulated functional connectivity matricies")


#Making the TPM based on the GIM

empirical_tpm_concat_sbys(time_course,output_path+"TPM_list")