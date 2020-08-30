import numpy as np
import scipy  

__all__ = ['ma_innovations']

from ... import operators

#------------------------------------------------------------------

def ma_innovations(x, order, mode = 'straight', unbias = False):
    ''''
    Moving Average (MA) model based 
        on the innovations of predictions method.  
    
    Parameters
    --------------
    * x: 1d ndarray,
        1-d input ndarray.
    * mode: string,
        mode of correlation function, 
        mode = {full, same, straight}.
    * order: int,
        is the order of the 
        desired moving-average model (zeros order).        
    * unbias: bool, 
        if True, unbiased covariance function will be taken.

    Returns
    ------------------
    * a: 1d ndarray (complex (or float)),
        auxilary autoregression coefficient (a=[1]). 
    * b: 1d ndarray (complex (or float)),
        moving-average coefficients of 
        the desired moving-average model.        
    * noise_variace: complex (or float), 
        variance of model residulas.

    Notes
    ---------
    
    See also
    ----------
    ma_dubrin,
    ma_yule_walker
    
    Examples
    -------------
    
    References
    ------------
    [1] Brockwell, P. J., & Davis, R. A. 
        Time series: Theory and methods (2nd ed.).
        New York: Springer, 1991.
    [2] https://github.com/statsmodels/statsmodels/blob/
        1212616d27ab820d303377f0bcf421cd3f46c289/statsmodels/
        tsa/innovations/_arma_innovations.pyx.in - statsmodels
        cython implementation.
        
        
    '''       
    x = np.asarray(x)
    N = x.shape[0]
        
    err   = np.zeros(order, dtype = x.dtype)
    bmat  = np.zeros((order, N), dtype = x.dtype)
    
    r = operators.correlation(x, mode = mode, unbias = unbias)
    
    r = np.asarray(r, dtype = x.dtype)

    err[0] = r[0]
    
    for i in np.arange(1, order):
        
        for k in np.arange(i):

            sub = np.sum(bmat[k, 1:k+1] * bmat[i, i-k+1:i+1] * err[:k][::-1])
            bmat[i, i - k] = 1. / err[k] * (r[i - k] - sub)

        err[i] = r[0] - np.sum(np.square(bmat[i, 1:i+1]) * err[:i][::-1])
 
    a = np.asarray([1])
    b = bmat[-1, :order]
    error = err[-1]

    return a, b, error
    
    
