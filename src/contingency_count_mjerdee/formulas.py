# This file contains python implementations of the "analytic formulas" for log Omega
from typing import Union, List
number = Union[int, float]

def logOmega_DM(n1: List[int],n2: List[int],alpha=1: number) -> number:
  """_summary_

  Args:
      n1 (List[int]): _description_
      n2 (List[int]): _description_
      alpha (_type_, optional): _description_. Defaults to 1:number.

  Returns:
      number: _description_
  """
  return n1

# Testing

import numpy as np
from math import log,exp
from numpy import loadtxt,zeros,sum
from numpy import log as nplog
from scipy.sparse import coo_matrix
from scipy.special import gamma, gammaln
import cvxpy as cp

def logBinom(a,b):
  if a > 0 and b > 0:
    return gammaln(a + 1) - gammaln(b + 1) - gammaln(a-b+1)
  return 0

def QMat(n1,n2,Z,alpha):
  q1 = len(n1)
  q2 = len(n2)
  Q = np.zeros((q1+q2-1,q1+q2-1))
  for r in range(q1):
    for s in range(q2 - 1):
      Q[r,s+q1] = Q[s+q1,r] = Z[r,s]**2/alpha + Z[r,s]
  for r in range(q1):
    Q[r,r] = n1[r] + np.sum(Z**2,axis=1)[r]/alpha
  for s in range(q2-1):
    Q[s+q1,s+q1] = n2[s] + np.sum(Z**2,axis=0)[s]/alpha
  return Q

# find the Z which minimizes (x+1)log(x+1) - x Log x
def findZ(n1,n2,alpha):
  X = cp.Variable(shape=(len(n1),len(n2)),pos=True) # X Variable
  gFunc = cp.sum(cp.log1p(X)-cp.rel_entr(X,X+alpha))
  constraints = [cp.sum(X, axis=1) == n1, cp.sum(X, axis=0) == n2,X >= np.zeros((len(n1),len(n2)))]
  prob = cp.Problem(cp.Maximize(gFunc),constraints)
  opt = prob.solve()
  return X.value

def gF(X,alpha):
  Xf = X.flatten()
  res = 0
  for x in Xf:
    res += (alpha + x)*log(alpha + x) - x*log(x) - alpha*log(alpha)
  return res

def QMat(n1,n2,Z,alpha):
  q1 = len(n1)
  q2 = len(n2)
  Q = np.zeros((q1+q2-1,q1+q2-1))
  for r in range(q1):
    for s in range(q2 - 1):
      Q[r,s+q1] = Q[s+q1,r] = Z[r,s]**2/alpha + Z[r,s]
  for r in range(q1):
    Q[r,r] = n1[r] + np.sum(Z**2,axis=1)[r]/alpha
  for s in range(q2-1):
    Q[s+q1,s+q1] = n2[s] + np.sum(Z**2,axis=0)[s]/alpha
  return Q

def logOmegaME_Gaussian(n1,n2,alpha):
  q1 = len(n1)
  q2 = len(n2)
  Z = findZ(n1,n2,alpha)
  g_Z = gF(Z,alpha)
  QL = QMat(n1,n2,Z,alpha)
  detQL = np.linalg.det(QL)
  Qinv = np.linalg.inv(QL)
  return g_Z  - (q1+q2-1)/2*np.log(2*np.pi)-1/2*np.log(detQL)

def logOmegaME_Edgeworth(n1,n2,alpha):
  q1 = len(n1)
  q2 = len(n2)
  Z = findZ(n1,n2,alpha)
  g_Z = gF(Z,alpha)
  QL = QMat(n1,n2,Z,alpha)
  detQL = np.linalg.det(QL)
  Qinv = np.linalg.inv(QL)
  gauss = g_Z -(q1+q2-1)/2*np.log(2*np.pi)-1/2*np.log(detQL)
 
  nu = 0
  for r in range(q1):
    for s in range(q2-1):
      nu += 1/24*Z[r,s]*(Z[r,s]+alpha)*(6*Z[r,s]**2+6*Z[r,s]*alpha+alpha**2)/alpha**3*3*(Qinv[r, r] + 2*Qinv[r, q1 + s] + Qinv[q1 + s, q1 + s])**2
  
  mu = 0
  for r1 in range(q1):
    for s1 in range(q2-1):
      for r2 in range(q1):
        for s2 in range(q2-1):
          mu += 1/36*Z[r1,s1]*(Z[r1,s1]+alpha)*(2*Z[r1,s1]+alpha)*Z[r2,s2]*(Z[r2,s2]+alpha)*(2*Z[r2,s2]+alpha)/alpha**4*3*(Qinv[r1, r2] + Qinv[r1, q1 + s2] + Qinv[r2, q1 + s1] + \
   Qinv[q1 + s1, q1 + s2])*(2*(Qinv[r1, r2] + Qinv[r1, q1 + s2] + \
      Qinv[r2, q1 + s1] + Qinv[q1 + s1, q1 + s2])**2 + \
   3*(Qinv[r1, r1] + 2*Qinv[r1, q1 + s1] + \
      Qinv[q1 + s1, q1 + s1])*(Qinv[r2, r2] + 2*Qinv[r2, q1 + s2] + \
      Qinv[q1 + s2, q1 + s2]))
  
  return gauss - mu/2 + nu

def fallingFactorial(x,k):
  result = 1
  for i in range(k):
    result *= (x - i)
  return result

def logOmegaEC(rs,cs,symmetrize=False):
  rs = np.array(rs)
  cs = np.array(cs)
  if symmetrize:
    return (logOmegaEC(rs,cs,symmetrize=False)+logOmegaEC(cs,rs,symmetrize=False))/2
  else:
    m = len(rs)
    N = np.sum(rs)
    alphaC = ((1-1/N)+(1-np.sum((cs/N)**2))/m)/(np.sum((cs/N)**2)-1/N)
    result = -logBinom(N + m*alphaC - 1, m*alphaC - 1)
    for r in rs:
      result += logBinom(r + alphaC - 1,alphaC-1)
    for c in cs:
      result += logBinom(c + m - 1, m - 1)
    return result

def logOmega0EC(rs,cs,symmetrize=False):
  rs = np.array(rs)
  cs = np.array(cs)
  if symmetrize:
    return (logOmegaEC(rs,cs,symmetrize=False)+logOmegaEC(cs,rs,symmetrize=False))/2
  else:
    m = len(rs)
    N = np.sum(rs)
    alphaC = ((1-1/N)+(1-np.sum((cs/N)**2))/m)/(np.sum((cs/N)**2)-1/N)
    result = -logBinom(m*alphaC,N)
    for r in rs:
      result += logBinom(alphaC,r)
    for c in cs:
      result += logBinom(m,c)
    return result

def logOmegaGM(rs,cs,symmetrize=False):
  rs = np.array(rs)
  cs = np.array(cs)
  if symmetrize:
    return (logOmegaGM(rs,cs,symmetrize=False)+logOmegaGM(cs,rs,symmetrize=False))/2
  else:
    m = len(rs)
    N = np.sum(rs)
    sigma2 = (np.sum(cs**2) + m*N)*(m - 1)/((m+1)*m**2)
    Q = (m-1)/(sigma2*m)*(np.sum(rs**2) - N**2/m)
    result = (m-1)/2*np.log((m-1)/(2*np.pi*m*sigma2)) + 1/2*np.log(m) - Q/2
    for c in cs:
      result += logBinom(c + m - 1, m - 1)
    return result

def logOmegaDE(rs,cs,symmetrize=False):
  rs = np.array(rs)
  cs = np.array(cs)
  if symmetrize:
    return (logOmegaDE(rs,cs,symmetrize=False)+logOmegaDE(cs,rs,symmetrize=False))/2
  else:
    N = np.sum(rs)
    m = len(rs)
    n = len(cs)
    w = N/(N+m*n/2)
    rbs = (1-w)/m + w*rs/N
    cbs = (1-w)/n + w*cs/N
    Kc = (m+1)/(m*np.sum(cbs**2)) - 1/m
    result = (m - 1)*(n - 1)*np.log(N + m*n/2) + gammaln(m*Kc) - n*gammaln(m) - m*gammaln(Kc)
    for rb in rbs:
      result += (Kc-1)*np.log(rb)
    for cb in cbs:
      result += (m-1)*np.log(cb)
    return result

def logOmegaBBK(rs,cs):
  rs = np.array(rs)
  cs = np.array(cs)
  N = np.sum(rs)
  return gammaln(N+1) - np.sum(gammaln(rs+1)) - np.sum(gammaln(cs+1)) \
  +2/N**2*np.sum(rs*(rs-1)/2)*np.sum(cs*(cs-1)/2)

def logOmega0BBK(rs,cs):
  rs = np.array(rs)
  cs = np.array(cs)
  N = np.sum(rs)
  return gammaln(N+1) - np.sum(gammaln(rs+1)) - np.sum(gammaln(cs+1)) \
  -2/N**2*np.sum(rs*(rs-1)/2)*np.sum(cs*(cs-1)/2)

def logOmegaGMW(rs,cs):
  rs = np.array(rs)
  cs = np.array(cs)
  N = np.sum(rs)
  R2 = C2 = R3 = C3 = 0
  for r in rs:
    R2 += fallingFactorial(r,2)
    R3 += fallingFactorial(r,3)
  for c in cs:
    C2 += fallingFactorial(c,2)
    C3 += fallingFactorial(c,3)
  return gammaln(N+1) - np.sum(gammaln(rs+1)) - np.sum(gammaln(cs+1)) \
  + R2*C2/(2*N**2) + R2*C2/(2*N**3) + R3*C3/(3*N**3) - R2*C2*(R2+C2)/(4*N**4)\
  - (R2**2*C3+R3*C2**2)/(2*N**4) + R2**2*C2**2/(2*N**5)
  
def logOmega0GMW(rs,cs):
  rs = np.array(rs)
  cs = np.array(cs)
  N = np.sum(rs)
  R2 = C2 = R3 = C3 = 0
  for r in rs:
    R2 += fallingFactorial(r,2)
    R3 += fallingFactorial(r,3)
  for c in cs:
    C2 += fallingFactorial(c,2)
    C3 += fallingFactorial(c,3)
  return gammaln(N+1) - np.sum(gammaln(rs+1)) - np.sum(gammaln(cs+1)) \
  - R2*C2/(2*N**2) - R2*C2/(2*N**3) + R3*C3/(3*N**3) - R2*C2*(R2+C2)/(4*N**4)\
  - (R2**2*C3+R3*C2**2)/(2*N**4) + R2**2*C2**2/(2*N**5)

def logOmegaGC(rs,cs):
  m = len(rs)
  n = len(cs)
  N = np.sum(rs)
  result = -logBinom(N + m*n - 1,N)
  for r in rs:
    result += logBinom(r + n - 1,r)
  for c in cs:
    result += logBinom(c + m - 1,c)
  return result

def logOmega0GC(rs,cs):
  m = len(rs)
  n = len(cs)
  N = np.sum(rs)
  result = -logBinom(m*n,N)
  for r in rs:
    result += logBinom(n,r)
  for c in cs:
    result += logBinom(m,c)
  return result

# Dense 0-1
def logOmega0CGM(rs,cs):
  m = len(rs)
  n = len(cs)
  N = np.sum(rs)
  R = np.sum((rs - N/m)**2)
  C = np.sum((cs - N/n)**2)
  l = N/(m*n)
  A = l*(1-l)/2
  result = -logBinom(m*n,N)
  for r in rs:
    result += logBinom(n,r)
  for c in cs:
    result += logBinom(m,c)
  result -= 1/2*(1 - R/(2*A*m*n))*(1 - C/(2*A*m*n))
  return result