import numpy as np
from math import log
from scipy.special import loggamma
import cvxpy as cp

########################################
# Linear-time estimates

def logBinom(a,b):
  if a > 0 and b > 0:
    if a-b+1 >= 0:
      return loggamma(a + 1) - loggamma(b + 1) - loggamma(a-b+1)
    else:
      return loggamma(a + 1) - loggamma(b + 1) - (loggamma(-(a-b+1))-np.pi*1.j)
  return 0
  

def fallingFactorial(x,k):
  result = 1
  for i in range(k):
    result *= (x - i)
  return result


def logOmega0EC(rs,cs,symmetrize=False):
  rs = np.array(rs)
  cs = np.array(cs)
  if symmetrize:
    return (logOmega0EC(rs,cs,symmetrize=False)+logOmega0EC(cs,rs,symmetrize=False))/2
  else:
    m = len(rs)
    N = np.sum(rs)
    alphaC = ((1-1/N)-(1-np.sum((cs/N)**2))/m)/(np.sum((cs/N)**2)-1/N)
    result = -logBinom(m*alphaC,N)
    for r in rs:
      result += logBinom(alphaC,r)
    for c in cs:
      result += logBinom(m,c)
    return result.real # Throwing away the absolute value at this stage is the same as taking the absolute value of \Omega_0


def logOmega0BBK(rs,cs):
  rs = np.array(rs)
  cs = np.array(cs)
  N = np.sum(rs)
  return loggamma(N+1) - np.sum(loggamma(rs+1)) - np.sum(loggamma(cs+1)) \
  -2/N**2*np.sum(rs*(rs-1)/2)*np.sum(cs*(cs-1)/2)


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
  return loggamma(N+1) - np.sum(loggamma(rs+1)) - np.sum(loggamma(cs+1)) \
  - (R2/N)*(C2/N)/2 - (R2/N)*(C2/N)/(2*N) + (R3/N)*(C3/N)/(3*N) - (R2/N)*(C2/N)*((R2+C2)/N)/(4*N)\
  - ((R2/N)**2*(C3/N)+(R3/N)*(C2/N)**2)/(2*N) + (R2/N)**2*(C2/N)**2/(2*N)


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


################################################
# Maximum-entropy estimates


def findZ(n1,n2,solver='ECOS',alpha=1):
  assert solver in ['ECOS','SCS'], str(solver)+' is not a supported solver'
  X = cp.Variable(shape=(len(n1),len(n2)),pos=True) # X Variable
  Xm = cp.Variable(shape=(len(n1),len(n2)),pos=True) # 1-X Variable (needed to establish concavity)
  hFunc = cp.sum(cp.entr(X)+cp.entr(Xm)) 
  constraints = [cp.sum(X, axis=1) == n1, cp.sum(X, axis=0) == n2,X >= np.zeros((len(n1),len(n2))),X <= np.ones((len(n1),len(n2))),Xm >= np.zeros((len(n1),len(n2))),Xm <= np.ones((len(n1),len(n2))),X + Xm == np.ones((len(n1),len(n2)))]
  prob = cp.Problem(cp.Maximize(hFunc),constraints)
  if solver == 'ECOS':
    opt = prob.solve(solver=cp.ECOS)
  if solver == 'SCS':
    opt = prob.solve(solver=cp.SCS)
  return X.value

def hF(X):
  Xf = X.flatten()
  res = 0
  for x in Xf:
    if x > 0 and x < 1:
      res += -(1-x)*log(1-x) - x*log(x) #(alpha + x)*log(alpha + x) - x*log(x) - alpha*log(alpha)
  return res

def QMat(n1,n2,Z):
  q1 = len(n1)
  q2 = len(n2)
  Q = np.zeros((q1+q2-1,q1+q2-1))
  for r in range(q1):
    for s in range(q2 - 1):
      Q[r,s+q1] = Q[s+q1,r] = Z[r,s] - Z[r,s]**2
  for r in range(q1):
    Q[r,r] = n1[r] - np.sum(Z**2,axis=1)[r]
  for s in range(q2-1):
    Q[s+q1,s+q1] = n2[s] - np.sum(Z**2,axis=0)[s]
  return Q

def removeAllZeroOnes(rs,cs):
  n = len(cs)
  temp_rs = []
  temp_cs = cs.copy()
  change = False
  for r in rs:
    if r == 0:
      change = True
    if r == n:
      change = True
      for i in range(len(temp_cs)):
        temp_cs[i] -= 1
    if r != 0 and r != n:
      temp_rs.append(r)
  m = len(temp_rs)
  temp_cs2 = []
  for c in temp_cs:
    if c == 0:
      change = True
    if c == m:
      change = True
      for i in range(len(temp_rs)):
        temp_rs[i] -= 1
    if c != 0 and c != m:
      temp_cs2.append(c)
  if change:
    return removeAllZeroOnes(temp_rs, temp_cs2)
  else:
    return temp_rs, temp_cs2


def logOmega0ME_Gaussian(rs, cs):
    rs, cs = removeAllZeroOnes(rs,cs)
    m = len(rs)
    n = len(cs)
    Z = findZ(rs,cs,solver='ECOS')
    h_Z = hF(Z)
    QL = QMat(rs,cs,Z)

    (sign, logdetQL) = np.linalg.slogdet(QL)
    return h_Z  - (n+m-1)/2*np.log(2*np.pi)-1/2*logdetQL


def logOmega0ME_Edgeworth(rs, cs):
    rs, cs = removeAllZeroOnes(rs,cs)
    m = len(rs)
    n = len(cs)
    Z = findZ(rs,cs,solver='ECOS')
    h_Z = hF(Z)
    QL = QMat(rs,cs,Z)

    (sign, logdetQL) = np.linalg.slogdet(QL)
    gaussResult =  h_Z  - (n+m-1)/2*np.log(2*np.pi)-1/2*logdetQL
    Qinv = np.linalg.inv(QL)
       
    nu = 0
    nu = np.sum(1/24*Z[:,:n-1]*(1-Z[:,:n-1])*(6*Z[:,:n-1]**2-6*Z[:,:n-1]+1)*3*(np.diagonal(Qinv)[:m][:, np.newaxis]+2*Qinv[:m,m:]+np.diagonal(Qinv)[m:])**2)
    
    mu = 0
    for r1 in range(m):
        mu_tensor = 1/36*Z[r1,:n-1,np.newaxis,np.newaxis]*(1-Z[r1,:n-1,np.newaxis,np.newaxis])*(2*Z[r1,:n-1,np.newaxis,np.newaxis]-1)*Z[np.newaxis,:,:n-1]*(1-Z[np.newaxis,:,:n-1])*(2*Z[np.newaxis,:,:n-1]-1)\
        *3*(np.swapaxes(Qinv[r1, :m,np.newaxis,np.newaxis],0,1) + np.swapaxes(Qinv[r1, m:,np.newaxis,np.newaxis],0,2) + np.swapaxes(Qinv[:m, m:,np.newaxis],0,1) + \
    np.swapaxes(Qinv[m:, m:,np.newaxis],1,2))*(2*(np.swapaxes(Qinv[r1, :m,np.newaxis,np.newaxis],0,1) + np.swapaxes(Qinv[r1, m:,np.newaxis,np.newaxis],0,2) \
        + np.swapaxes(Qinv[np.newaxis,:m, m:],0,2) + np.swapaxes(Qinv[m:, m:,np.newaxis],1,2))**2 + 3*(Qinv[r1, r1,np.newaxis,np.newaxis,np.newaxis] + 2*Qinv[r1, m:,np.newaxis,np.newaxis] + \
        np.diagonal(Qinv)[m:,np.newaxis,np.newaxis])*(np.swapaxes(np.diagonal(Qinv)[:m,np.newaxis,np.newaxis],0,1) + 2*Qinv[np.newaxis,:m, m:] + \
        np.swapaxes(np.diagonal(Qinv)[m:,np.newaxis,np.newaxis],0,2)))
        mu += np.sum(mu_tensor)

    return gaussResult - mu/2 + nu