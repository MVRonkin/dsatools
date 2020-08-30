import numpy as np
import scipy

from ._kernel_matrix import kernel_matrix

#-----------------------------------
def kernel_transform(x, kernel='linear', 
                     kpar=1, mode = 'toeplitz'):
    '''
    Kernel_basded transformation.
     
    Parameters 
    -----------------------
    * x: 1d ndarray.   
    * kernel: string,
        kernel type (linear for defoult similar as
         traditional covariance function), 
         kernals = {linear, rbf, thin_plate, sigmoid, 
                 poly, bump, polymorph, euclid}.
    * kpar: float,
        kernel parameter.
    * mode: string,
        covariance matrix mode,
        mode = {full,covar,traj,toeplitz}.
    
    Returns
    ---------- 
    * periodogram estimation 1d ndarray.

    References
    ------------ 
    
    Example
    ----------- 
    
    See also
    ---------

    '''
    x = np.asarray(x)
    N = x.shape[0]
    
    lags = N//2

    R = kernel_matrix(x,
                      mode=mode,
                      kernel=kernel,
                      kpar=kpar,
                      lags=lags+1,
                      ret_base=False,)
        
    rt = R[0,1:lags+1] 

    r = np.concatenate((np.conj(rt[::-1]),[R[0,0]], rt[:-1]))
    
    r /=N

    return r