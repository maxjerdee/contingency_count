# contingency_count

Linear-time and Maximum-Entropy estimates of estimating the number of contingency tables are given in the file log_Omega_estimates.py

Analogous estimates for the number of 0-1 tables are given in log_Omega_0_estimates.py

c++ code implementing the SIS method for contingency tables is given in log_Omega_SIS.cpp which can be compiled as
g++ -std=c++17 -o log_Omega_SIS log_Omega_SIS.cpp 

A typical usage of the compiled script would be:
./log_Omega_SIS -i in.txt -o out.txt -t 10000000

where in.txt contains the row and column margins on separate lines, space-delimited, and the SIS results will be output to the file out.txt. 
The -t flag indicates the maximum number of iterations to perform.

Similar c++ code of the SIS method for 0-1 tables is given in log_Omega_0_SIS.cpp, which can be similarly used as:
./log_Omega_0_SIS -i in.txt -o out.txt -t 10000000 -H CGM
The new -H flag can be used to specify what estimate of logOmega0 is used to generate tables. We recommend using CGM, but ECC and GC are available. 
