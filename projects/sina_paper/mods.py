import os,sys
import numpy as np



def dec2bin(num,num_digits):
    frmt = '{:0'
    frmt += str(num_digits)
    frmt += 'b}'
    return frmt.format(num)


# def dec2bin(string_num):
#     num = int(string_num)
#     mid = []
#     while True:
#         if num == 0: break
#         num,rem = divmod(num, 2)
#         mid.append(base[rem])
#
#     return ''.join([str(x) for x in mid[::-1]])