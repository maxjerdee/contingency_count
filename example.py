# This file gives examples of various methods used to estimate contingency tables on test cases

from contingency_count import log_Omega_estimates, wrappers
from subprocess import Popen, PIPE
from threading import Timer
import sys
import time
import json
import re
import numpy as np

example_cases = [{"title":"Small dense example (N = 592, m = n = 4):","rs":[220,215,93,64],"cs":[108,286,71,127],"verification":"lattE","truth":"34.7425=log(1225914276768514)"},\
    {"title":"Small sparse example (N = 80, m = n = 32):","rs":[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],"cs":[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],"verification":"Exact recursion","truth":"185.0300=log(227775532244174238018303070203887605268008568621335728477674638800921717534720000)"},\
    {"title":"Large dense example (N = 6400, m = n = 32):","rs":[53, 240, 454, 108, 484, 872, 279, 130, 8, 227, 264, 33, 68, 136, 228, 73, 225, 393, 173, 126, 27, 98, 162, 63, 82, 467, 332, 38, 222, 23, 177, 135],\
        "cs":[88, 512, 42, 857, 117, 225, 22, 164, 415, 76, 88, 56, 19, 204, 201, 16, 116, 901, 47, 10, 526, 74, 151, 312, 65, 494, 58, 70, 17, 292, 1, 164],"verification":"SIS, 1 hour","truth":"1973.66+/-0.05"},\
            {"title":"Large sparse example (N = 400, m = n = 128):","rs":[1, 1, 4, 7, 7, 3, 1, 1, 1, 1, 4, 3, 5, 6, 5, 2, 5, 1, 2, 1, 1, 2, 2, \
3, 2, 1, 1, 2, 1, 11, 7, 1, 1, 2, 3, 2, 2, 3, 2, 2, 8, 2, 5, 3, 1, 1, \
2, 1, 1, 2, 3, 1, 2, 4, 2, 2, 2, 2, 3, 9, 7, 2, 4, 1, 1, 6, 8, 2, 2, \
2, 5, 2, 2, 7, 8, 1, 1, 4, 5, 3, 4, 7, 1, 1, 1, 8, 3, 3, 4, 1, 6, 1, \
16, 5, 9, 2, 1, 1, 3, 3, 1, 1, 1, 1, 3, 1, 3, 4, 1, 1, 2, 1, 1, 2, 4, \
1, 4, 3, 2, 10, 2, 6, 4, 7, 4, 2, 5, 2],\
        "cs":[1, 2, 1, 6, 2, 5, 1, 3, 1, 1, 1, 2, 2, 4, 3, 14, 12, 11, 3, 1, 4, 6, \
1, 8, 1, 8, 4, 2, 1, 7, 1, 3, 2, 4, 3, 1, 3, 5, 3, 11, 2, 4, 3, 2, 1, \
3, 2, 1, 5, 1, 4, 9, 5, 1, 2, 5, 1, 1, 7, 1, 3, 1, 8, 1, 2, 2, 4, 4, \
1, 4, 2, 1, 1, 8, 1, 8, 4, 4, 2, 2, 1, 3, 3, 1, 2, 1, 1, 1, 1, 4, 5, \
3, 2, 1, 3, 1, 3, 6, 5, 9, 3, 2, 1, 1, 2, 1, 1, 1, 2, 4, 1, 1, 1, 3, \
3, 1, 4, 3, 4, 3, 1, 1, 8, 4, 1, 5, 1, 4]\
            ,"verification":"SIS, 1 hour","truth":"1316.4513+/-0.0003"}]
#  1316.4513811660304, 0.0002652289024354104, 
# 1316.4513+/-0.0003

for case in example_cases:
    print(case["title"])
    rs = case["rs"]
    cs = case["cs"]
    print("Row sums:",case["rs"])
    print("Column sums:",case["cs"])
    print("Exact Result ("+case["verification"]+"): "+case["truth"])
    print()
    print("Linear time estimates:")
    print(f"EC (our) Estimate: {log_Omega_estimates.logOmegaEC(rs,cs):.5f}")
    print(f"Gail & Mantel (1977): {log_Omega_estimates.logOmegaGM(rs,cs):.5f}")
    print(f"Diaconis & Efron (1985): {log_Omega_estimates.logOmegaDE(rs,cs):.5f}")
    print(f"Good & Crook (1977): {log_Omega_estimates.logOmegaGC(rs,cs):.5f}")
    print(f"Bekessy (1972): {log_Omega_estimates.logOmegaBBK(rs,cs):.5f}")
    print(f"Greenhill & McKay (2008): {log_Omega_estimates.logOmegaGMK(rs,cs):.5f}")
    print()
    print("Maximum entropy estimates:")
    print(f"Maximum-entropy, Gaussian: {log_Omega_estimates.logOmegaME_Gaussian(rs,cs):.5f}")
    print(f"Maximum-entropy, Edgeworth: {log_Omega_estimates.logOmegaME_Edgeworth(rs,cs):.5f}")
    if case["title"][0] == "S": # Results only quickly converge for our small cases
        print()
        print("SIS methods (10 seconds each):")
        SIS_value, SIS_error = wrappers.logOmega_SIS(rs,cs,time=10,mode="EC")
        print(f'SIS (EC-based): {SIS_value:.5f}+/-{SIS_error:.5f}')
        SIS_value, SIS_error = wrappers.logOmega_SIS(rs,cs,time=10,mode="GC")
        print(f'SIS (GC-based): {SIS_value:.5f}+/-{SIS_error:.5f}')
    print()
    print()