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


def logOmegaGMW(rs, cs):
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

def QMat(n1, n2, Z, alpha=1):
    q1 = len(n1)
    q2 = len(n2)
    Q = np.zeros((q1+q2-1, q1+q2-1))
    for r in range(q1):
        for s in range(q2 - 1):
            Q[r, s+q1] = Q[s+q1, r] = Z[r, s]**2/alpha + Z[r, s]
    for r in range(q1):
        Q[r, r] = n1[r] + np.sum(Z**2, axis=1)[r]/alpha
    for s in range(q2-1):
        Q[s+q1, s+q1] = n2[s] + np.sum(Z**2, axis=0)[s]/alpha
    return Q


def findZ(n1, n2, solver='ECOS', alpha=1):
    assert solver in ['ECOS', 'SCS'], str(solver)+' is not a supported solver'
    X = cp.Variable(shape=(len(n1), len(n2)), pos=True)  # X Variable
    gFunc = cp.sum(cp.log1p(X)-cp.rel_entr(X, X+alpha))
    constraints = [cp.sum(X, axis=1) == n1, cp.sum(
        X, axis=0) == n2, X >= np.zeros((len(n1), len(n2)))]
    prob = cp.Problem(cp.Maximize(gFunc), constraints)
    if solver == 'ECOS':
        opt = prob.solve(solver=cp.ECOS)
    if solver == 'SCS':
        opt = prob.solve(solver=cp.SCS)
    return X.value


def gF(X, alpha=1):
    Xf = X.flatten()
    res = 0
    for x in Xf:
        res += (alpha + x)*log(alpha + x) - x*log(x) - alpha*log(alpha)
    return res


def QMat(n1, n2, Z, alpha=1):
    q1 = len(n1)
    q2 = len(n2)
    Q = np.zeros((q1+q2-1, q1+q2-1))
    for r in range(q1):
        for s in range(q2 - 1):
            Q[r, s+q1] = Q[s+q1, r] = Z[r, s]**2/alpha + Z[r, s]
    for r in range(q1):
        Q[r, r] = n1[r] + np.sum(Z**2, axis=1)[r]/alpha
    for s in range(q2-1):
        Q[s+q1, s+q1] = n2[s] + np.sum(Z**2, axis=0)[s]/alpha
    return Q


def logOmegaME_Gaussian(n1, n2, alpha=1):
    q1 = len(n1)
    q2 = len(n2)
    Z = findZ(n1, n2, solver='ECOS', alpha=alpha) # The solver SCS can also be used, although in practice it seems that the 
    g_Z = gF(Z, alpha)
    QL = QMat(n1, n2, Z, alpha)
    (sign, logdetQL) = np.linalg.slogdet(QL)
    Qinv = np.linalg.inv(QL)
    return g_Z - (q1+q2-1)/2*np.log(2*np.pi)-1/2*logdetQL


def logOmegaME_Edgeworth(n1, n2, alpha=1):
    q1 = len(n1)
    q2 = len(n2)
    Z = findZ(n1, n2, solver='ECOS', alpha=alpha)
    g_Z = gF(Z, alpha)
    QL = QMat(n1, n2, Z, alpha)
    (sign, logdetQL) = np.linalg.slogdet(QL)
    Qinv = np.linalg.inv(QL)
    gauss = g_Z - (q1+q2-1)/2*np.log(2*np.pi)-1/2*logdetQL

    nu = np.sum(1/24*Z[:, :q2-1]*(Z[:, :q2-1]+alpha)*(6*Z[:, :q2-1]**2+6*Z[:, :q2-1]*alpha+alpha**2) /
                alpha**3*3*(np.diagonal(Qinv)[:q1][:, np.newaxis]+2*Qinv[:q1, q1:]+np.diagonal(Qinv)[q1:])**2)

    mu_tensor = 1/36*Z[:, :q2-1, np.newaxis, np.newaxis]*(Z[:, :q2-1, np.newaxis, np.newaxis]+alpha)*(2*Z[:, :q2-1, np.newaxis, np.newaxis]+alpha)*Z[np.newaxis, np.newaxis, :, :q2-1]*(Z[np.newaxis, np.newaxis, :, :q2-1]+alpha)*(2*Z[np.newaxis, np.newaxis, :, :q2-1]+alpha)/alpha**4*3 *\
        (np.swapaxes(Qinv[:q1, :q1, np.newaxis, np.newaxis], 1, 2) + np.swapaxes(Qinv[:q1, q1:, np.newaxis, np.newaxis], 1, 3) +
         np.swapaxes(Qinv[:q1, q1:, np.newaxis, np.newaxis], 0, 2)+np.swapaxes(Qinv[np.newaxis, np.newaxis, q1:, q1:], 1, 2))\
        * (2*(np.swapaxes(Qinv[:q1, :q1, np.newaxis, np.newaxis], 1, 2) + np.swapaxes(Qinv[:q1, q1:, np.newaxis, np.newaxis], 1, 3) +
              np.swapaxes(Qinv[:q1, q1:, np.newaxis, np.newaxis], 0, 2) + np.swapaxes(Qinv[np.newaxis, np.newaxis, q1:, q1:], 1, 2))**2
           + 3*(np.diagonal(Qinv)[:q1, np.newaxis, np.newaxis, np.newaxis] + 2*Qinv[:q1, q1:, np.newaxis, np.newaxis]+np.diagonal(Qinv)[np.newaxis, q1:, np.newaxis, np.newaxis]) *
           (np.diagonal(Qinv)[np.newaxis, np.newaxis, :q1, np.newaxis] + 2*Qinv[np.newaxis, np.newaxis, :q1, q1:] + np.diagonal(Qinv)[np.newaxis, np.newaxis, np.newaxis, q1:]))

    mu = np.sum(mu_tensor)

    # Loop Version:
    # nu = 0
    # for r in range(q1):
    #   for s in range(q2-1):
    #     nu += 1/24*Z[r,s]*(Z[r,s]+alpha)*(6*Z[r,s]**2+6*Z[r,s]*alpha+alpha**2)/alpha**3*3*(Qinv[r, r] + 2*Qinv[r, q1 + s] + Qinv[q1 + s, q1 + s])**2
    # mu = 0
    # for r1 in range(q1):
    #   for s1 in range(q2-1):
    #     for r2 in range(q1):
    #       for s2 in range(q2-1):
    #         mu += 1/36*Z[r1,s1]*(Z[r1,s1]+alpha)*(2*Z[r1,s1]+alpha)*Z[r2,s2]*(Z[r2,s2]+alpha)*(2*Z[r2,s2]+alpha)/alpha**4*3*(Qinv[r1, r2] + Qinv[r1, q1 + s2] + Qinv[r2, q1 + s1] + \
    #  Qinv[q1 + s1, q1 + s2])*(2*(Qinv[r1, r2] + Qinv[r1, q1 + s2] + \
    #     Qinv[r2, q1 + s1] + Qinv[q1 + s1, q1 + s2])**2 + \
    #  3*(Qinv[r1, r1] + 2*Qinv[r1, q1 + s1] + \
    #     Qinv[q1 + s1, q1 + s1])*(Qinv[r2, r2] + 2*Qinv[r2, q1 + s2] + \
    #     Qinv[q1 + s2, q1 + s2]))

    return gauss - mu/2 + nu


