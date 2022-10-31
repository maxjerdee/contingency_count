import numpy as np
from math import log
from scipy.special import gammaln
import cvxpy as cp

#########################################################
# Linear-time estimates

def logBinom(a, b):
    if a > 0 and b > 0:
        return gammaln(a + 1) - gammaln(b + 1) - gammaln(a-b+1)
    return 0


def fallingFactorial(x, k):
    result = 1
    for i in range(k):
        result *= (x - i)
    return result


def logOmegaEC(rs, cs, symmetrize=False):
    rs = np.array(rs)
    cs = np.array(cs)
    if symmetrize:
        return (logOmegaEC(rs, cs, symmetrize=False)+logOmegaEC(cs, rs, symmetrize=False))/2
    else:
        m = len(rs)
        N = np.sum(rs)
        alphaC = ((1-1/N)+(1-np.sum((cs/N)**2))/m)/(np.sum((cs/N)**2)-1/N)
        result = -logBinom(N + m*alphaC - 1, m*alphaC - 1)
        for r in rs:
            result += logBinom(r + alphaC - 1, alphaC-1)
        for c in cs:
            result += logBinom(c + m - 1, m - 1)
        return result


def logOmegaGM(rs, cs, symmetrize=False):
    rs = np.array(rs)
    cs = np.array(cs)
    if symmetrize:
        return (logOmegaGM(rs, cs, symmetrize=False)+logOmegaGM(cs, rs, symmetrize=False))/2
    else:
        m = len(rs)
        N = np.sum(rs)
        sigma2 = (np.sum(cs**2) + m*N)*(m - 1)/((m+1)*m**2)
        Q = (m-1)/(sigma2*m)*(np.sum(rs**2) - N**2/m)
        result = (m-1)/2*np.log((m-1)/(2*np.pi*m*sigma2)) + 1/2*np.log(m) - Q/2
        for c in cs:
            result += logBinom(c + m - 1, m - 1)
        return result


def logOmegaDE(rs, cs, symmetrize=False):
    rs = np.array(rs)
    cs = np.array(cs)
    if symmetrize:
        return (logOmegaDE(rs, cs, symmetrize=False)+logOmegaDE(cs, rs, symmetrize=False))/2
    else:
        N = np.sum(rs)
        m = len(rs)
        n = len(cs)
        w = N/(N+m*n/2)
        rbs = (1-w)/m + w*rs/N
        cbs = (1-w)/n + w*cs/N
        Kc = (m+1)/(m*np.sum(cbs**2)) - 1/m
        result = (m - 1)*(n - 1)*np.log(N + m*n/2) + \
            gammaln(m*Kc) - n*gammaln(m) - m*gammaln(Kc)
        for rb in rbs:
            result += (Kc-1)*np.log(rb)
        for cb in cbs:
            result += (m-1)*np.log(cb)
        return result


def logOmegaBBK(rs, cs):
    rs = np.array(rs)
    cs = np.array(cs)
    N = np.sum(rs)
    return gammaln(N+1) - np.sum(gammaln(rs+1)) - np.sum(gammaln(cs+1)) \
        + 2/N**2*np.sum(rs*(rs-1)/2)*np.sum(cs*(cs-1)/2)


def logOmegaGMK(rs, cs):
    rs = np.array(rs)
    cs = np.array(cs)
    N = np.sum(rs)
    R2 = C2 = R3 = C3 = 0
    for r in rs:
        R2 += fallingFactorial(r, 2)
        R3 += fallingFactorial(r, 3)
    for c in cs:
        C2 += fallingFactorial(c, 2)
        C3 += fallingFactorial(c, 3)
    return gammaln(N+1) - np.sum(gammaln(rs+1)) - np.sum(gammaln(cs+1)) \
        + (R2/N)*(C2/N)/2 + (R2/N)*(C2/N)/(2*N) + (R3/N)*(C3/N)/(3*N) - (R2/N)*(C2/N)*((R2+C2)/N)/(4*N)\
        - ((R2/N)**2*(C3/N)+(R3/N)*(C2/N)**2)/(2*N) + (R2/N)**2*(C2/N)**2/(2*N)


def logOmegaGC(rs, cs):
    m = len(rs)
    n = len(cs)
    N = np.sum(rs)
    result = -logBinom(N + m*n - 1, N)
    for r in rs:
        result += logBinom(r + n - 1, r)
    for c in cs:
        result += logBinom(c + m - 1, c)
    return result


def logOmega0GC(rs, cs):
    m = len(rs)
    n = len(cs)
    N = np.sum(rs)
    result = -logBinom(m*n, N)
    for r in rs:
        result += logBinom(n, r)
    for c in cs:
        result += logBinom(m, c)
    return result


###################################################3
# Maximum-entropy methods

def QMat(rs, cs, Z):
    m = len(rs)
    n = len(cs)
    Q = np.zeros((m+n-1, m+n-1))
    for r in range(m):
        for s in range(n - 1):
            Q[r, s+m] = Q[s+m, r] = Z[r, s]**2 + Z[r, s]
    for r in range(m):
        Q[r, r] = rs[r] + np.sum(Z**2, axis=1)[r]
    for s in range(n-1):
        Q[s+m, s+m] = cs[s] + np.sum(Z**2, axis=0)[s]
    return Q


def findZ(rs, cs, solver='ECOS'):
    assert solver in ['ECOS', 'SCS'], str(solver)+' is not a supported solver'
    X = cp.Variable(shape=(len(rs), len(cs)), pos=True)  # X Variable
    gFunc = cp.sum(cp.log1p(X)-cp.rel_entr(X, X+1))
    constraints = [cp.sum(X, axis=1) == rs, cp.sum(
        X, axis=0) == cs, X >= np.zeros((len(rs), len(cs)))]
    prob = cp.Problem(cp.Maximize(gFunc), constraints)
    if solver == 'ECOS':
        opt = prob.solve(solver=cp.ECOS)
    if solver == 'SCS':
        opt = prob.solve(solver=cp.SCS)
    return X.value


def gF(X):
    Xf = X.flatten()
    res = 0
    for x in Xf:
        res += (x+1)*log(x+1) - x*log(x)
    return res


def QMat(rs, cs, Z):
    m = len(rs)
    n = len(cs)
    Q = np.zeros((m+n-1, m+n-1))
    for r in range(m):
        for s in range(n - 1):
            Q[r, s+m] = Q[s+m, r] = Z[r, s]**2 + Z[r, s]
    for r in range(m):
        Q[r, r] = rs[r] + np.sum(Z**2, axis=1)[r]
    for s in range(n-1):
        Q[s+m, s+m] = cs[s] + np.sum(Z**2, axis=0)[s]
    return Q


def logOmegaME_Gaussian(rs, cs):
    m = len(rs)
    n = len(cs)
    Z = findZ(rs, cs, solver='ECOS') # The solver SCS can also be used, although in practice it seems that the 
    g_Z = gF(Z)
    QL = QMat(rs, cs, Z)
    (sign, logdetQL) = np.linalg.slogdet(QL)
    return g_Z - (m+n-1)/2*np.log(2*np.pi)-1/2*logdetQL


def logOmegaME_Edgeworth(rs, cs):
    m = len(rs)
    n = len(cs)
    Z = findZ(rs, cs, solver='ECOS')
    g_Z = gF(Z)
    QL = QMat(rs, cs, Z)
    (sign, logdetQL) = np.linalg.slogdet(QL)
    Qinv = np.linalg.inv(QL)
    corr = np.zeros((m+n,m+n))
    corr[:m+n-1,:m+n-1] = Qinv
    gauss = g_Z - (m+n-1)/2*np.log(2*np.pi)-1/2*logdetQL

    nu = np.sum(1/24*Z[:, :]*(Z[:, :]+1)*(6*Z[:, :]**2+6*Z[:, :]+1)*3*(np.diagonal(corr)[:m][:, np.newaxis]+2*corr[:m, m:]+np.diagonal(corr)[m:])**2)

    mu = 0
    for r1 in range(m):
        mu_tensor = 1/36*Z[r1, :, np.newaxis, np.newaxis]*(Z[r1, :, np.newaxis, np.newaxis]+1)*(2*Z[r1, :, np.newaxis, np.newaxis]+1)*Z[np.newaxis, :, :]*(Z[np.newaxis, :, :]+1)*(2*Z[np.newaxis, :, :]+1)*3 *\
            (np.swapaxes(corr[r1, :m, np.newaxis, np.newaxis], 0, 1) + np.swapaxes(corr[r1, m:, np.newaxis, np.newaxis], 0, 2) +
            np.swapaxes(corr[:m, m:, np.newaxis], 0, 1)+np.swapaxes(corr[np.newaxis, m:, m:], 0, 1))\
            * (2*(np.swapaxes(corr[r1, :m, np.newaxis, np.newaxis], 0, 1) + np.swapaxes(corr[r1, m:, np.newaxis, np.newaxis], 0, 2) +
                np.swapaxes(corr[:m, m:, np.newaxis], 0, 1) + np.swapaxes(corr[np.newaxis, m:, m:], 0, 1))**2
            + 3*(np.diagonal(corr)[r1, np.newaxis, np.newaxis, np.newaxis] + 2*corr[r1, m:, np.newaxis, np.newaxis]+np.diagonal(corr)[m:, np.newaxis, np.newaxis]) *
            (np.diagonal(corr)[np.newaxis, :m, np.newaxis] + 2*corr[np.newaxis, :m, m:] + np.diagonal(corr)[np.newaxis, np.newaxis, m:]))
        mu += np.sum(mu_tensor)

    # Loop Version:
    # nu = 0
    # for r in range(m):
    #   for s in range(n):
    #     nu += 1/24*Z[r,s]*(Z[r,s]+1)*(6*Z[r,s]**2+6*Z[r,s]+1)*3*(corr[r, r] + 2*corr[r, m + s] + corr[m + s, m + s])**2
    # mu = 0
    # for r1 in range(m):
    #   for s1 in range(n):
    #     for r2 in range(m):
    #       for s2 in range(n):
    #         mu += 1/36*Z[r1,s1]*(Z[r1,s1]+1)*(2*Z[r1,s1]+1)*Z[r2,s2]*(Z[r2,s2]+1)*(2*Z[r2,s2]+1)*3*(corr[r1, r2] + corr[r1, m + s2] + corr[r2, m + s1] + \
    #  corr[m + s1, m + s2])*(2*(corr[r1, r2] + corr[r1, m + s2] + \
    #     corr[r2, m + s1] + corr[m + s1, m + s2])**2 + \
    #  3*(corr[r1, r1] + 2*corr[r1, m + s1] + \
    #     corr[m + s1, m + s1])*(corr[r2, r2] + 2*corr[r2, m + s2] + \
    #     corr[m + s2, m + s2]))

    return gauss - mu/2 + nu


