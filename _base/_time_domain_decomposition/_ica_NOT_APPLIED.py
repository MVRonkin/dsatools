import numpy as np
import scipy  

from ... import spectrum
from ... import utilits as ut


def ica_kurtosis(x, order, mode = 'full'):
    '''
    FUNCTION IN TEST
    
    Max-kurtosis Independent Component Analysis (ICA)
    
    References
    ------------------------
    [1] http://www.cs.nyu.edu/~roweis/kica.html
    
    '''
    X = signals.matrix.kernel_martix(x, mode=mode,  ktype='linear', kpar=0.001, lags = x.size//2)

    invCov = np.linalg.inv(X.T.dot(np.conj(X)))
    W = scipy.linalg.sqrtm(invCov)
    Xcw = np.dot(W , X)
       
    gg = repmat(np.sum(np.square(Xcw),axis=1), Xcw.shape[0], 1)
    TEST= np.dot(gg*Xcw, Xcw.T)
    es,ev = np.linalg.eig(TEST)
    Zica = np.dot(ev[:order,:], Xcw)

    return Zica

def repmat(a, m, n):

    a = np.asanyarray(a)
    ndim = a.ndim
    if ndim == 0:
        origrows, origcols = (1, 1)
    elif ndim == 1:
        origrows, origcols = (1, a.shape[0])
    else:
        origrows, origcols = a.shape
    rows = origrows * m
    cols = origcols * n
    c = a.reshape(1, a.size).repeat(m, 0).reshape(rows, origcols).repeat(n, 0)
    return c.reshape(rows, cols)