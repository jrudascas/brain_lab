import numpy as np
from projects.sina_paper.mods import dec2bin


# Generates an array of 0s and 1s which represent all possible configurations of
# the system

def gen_reservoir(N):
    # list turns this into integers
    M = np.array([list(dec2bin(x, N))[::-1] for x in range(0, 2 ** N)])
    return M