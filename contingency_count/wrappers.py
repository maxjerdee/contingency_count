import subprocess
from os.path import join
import os
import numpy as np
import re
import json
import tempfile 

def log_from_string(value_string,error_string):
    offset = int(re.findall('E10\+[0-9]+',value_string)[0][4:])
    value = np.log(float(re.findall('.*E10\+',value_string)[0][:-4])) + offset*np.log(10)
    lower_value = np.log(float(re.findall('.*E10\+',value_string)[0][:-4])-float(re.findall('.*E10\+',error_string)[0][:-4])) + offset*np.log(10)
    return value, value - lower_value

def logOmega_SIS(rs,cs,time=60,max_iters=10000000,mode="EC"):
  # Use temporary file to feed in the data (ideally I would bind this properly)
  with tempfile.NamedTemporaryFile(mode='w',dir="./temp_files",delete=False) as fp:
    for num in rs:
      fp.write(str(num)+' ')
    fp.write('\n')
    for num in cs:
      fp.write(str(num)+' ')
    fp.write('\n')
    fp.close()

    completed_process = subprocess.run(["./contingency_count/log_Omega_SIS", "-i",fp.name,"-w","F","-t",str(max_iters),"-T",str(time),"-M",mode],check=True,capture_output=True)
    os.remove(fp.name)
    # Read data
    data = completed_process.stdout
    output_dict = json.loads(data.decode())

    value,error = log_from_string(output_dict['value'],output_dict['error'])
    return value,error

## IGNORE BEYOND HERE, work in progress

# We provide rough wrappers for the executables which implement the lattE program, 
# Harrison & Miller's exact recursion method, and the SIS methods described in the paper
# These wrappers make use of temporary files, and so should not be multithreaded or they may
# interfere with each others' operation
# This are also Linux executables, and so will not run on Windows or macOS

# # Wrapper for the lattE executable in this context. Wrapper will operate within the lattE_wrap directory
# def run_lattE(rs,cs):
#     lattE_wrap_folder = 'lattE_wrap'
#     # Write the rs and cs into a format which lattE can read:
#     print(rs,cs)
#     n = np.sum(rs)
#     m = len(rs)
#     n = len(cs)
#     num_constraints = m + n - 1
#     num_vars = (m-1)*(n-1)
#     f = open(join(lattE_wrap_folder,'input.txt'), "w")
#     # write dimensions
#     f.write(str(num_constraints+num_vars) + " "+ str(num_vars + 1))
#     f.write('\n')
#     # write marginalization constraints
#     for r in range(m - 1):
#       f.write(str(rs[r]) + " ")
#       for rt in range(m - 1):
#         for st in range(n - 1):
#           if r == rt:
#             f.write("-1 ")
#           else:
#             f.write("0 ")
#       f.write('\n')
#     for s in range(n - 1):
#       f.write(str(cs[s])+" ")
#       for rt in range(m - 1):
#         for st in range(n - 1):
#           if s == st:
#             f.write("-1 ")
#           else:
#             f.write("0 ")
#       f.write('\n')
#     f.write(str(rs[-1]+cs[-1]-n)+" ")
#     # write non-negativity constraints
#     for rt in range(m - 1):
#       for st in range(n - 1):
#         f.write("1 ")
#     f.write('\n')
#     for i in range(num_vars):
#       for j in range(num_vars+1):
#         if i + 1 == j:
#           f.write("1 ")
#         else:
#           f.write("0 ")
#       f.write('\n')
      
#     completed_process = subprocess.run(["./lattE_wrap/count_lattE", "./lattE_wrap/input.txt"],check=True,capture_output=True)
#     completed_process.stdout