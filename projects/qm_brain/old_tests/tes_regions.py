import numpy as np

occipital_R_ind = [61 ,56 ,57 ,62 ,66 ,63 ,67 ,70]
occipital_C_ind = [54 ,55]
occipital_L_ind = [52, 53, 51, 50, 49, 46, 45, 44]

parietal_R_ind = [58 ,59, 60 ,65 ,64 ,68 ,69 ,77 ,72 ,71 ,74 ,75 ,76 ,81 ,80 ,85]
parietal_C_ind = [48, 42]
parietal_L_ind = [47 ,41 ,24 ,29 ,40 ,39 ,33 ,28 ,36 ,38 ,37 ,35 ,32 ,31 ,34 ,30]

posteriorf_R_ind = [79 ,78 ,84 ,3 ,88 ,83 ,82 ,87 ,91 ,86]
posteriorf_C_ind = [4]
posteriorf_L_ind = [5 ,23 ,10 ,9 ,15 ,22 ,27 ,21 ,18 ,26]

anteriorf_R_ind = [2 ,7 ,6 ,1 ,90 ,0 ,89]
anteriorf_C_ind = [8 ,12 ,11]
anteriorf_L_ind = [14 ,13 ,16 ,17 ,20 ,19 ,25]

sum_list = [occipital_R_ind, occipital_C_ind, occipital_L_ind,
            parietal_R_ind, parietal_C_ind, parietal_L_ind,
            posteriorf_R_ind, posteriorf_C_ind, posteriorf_L_ind,
            anteriorf_R_ind, anteriorf_C_ind, anteriorf_L_ind]

sumo = occipital_R_ind + occipital_C_ind + occipital_L_ind + parietal_R_ind + parietal_C_ind + parietal_L_ind + posteriorf_R_ind + posteriorf_C_ind + posteriorf_L_ind + anteriorf_R_ind + anteriorf_C_ind + anteriorf_L_ind

sum_arr = np.sum(np.array(sumo))

sum_elec = np.sum(np.arange(92))

print(sum_arr)
print(sum_elec)

print(len(sumo))