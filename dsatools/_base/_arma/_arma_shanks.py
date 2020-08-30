import numpy as np
import scipy  

from ... import operators
from ... import utilits as ut

from ._arma_prony import arma_prony

__all__ = ['arma_shanks']
#-----------------------------------------------------------
def arma_shanks(x, poles_order, zeros_order):
    '''
    Autoregression-Moving Average (ARMA) model based on the 
        Shanks signal model.  
    
    Parameters
    ----------------------
    * x:  1d ndarray. 
    * poles_order: int.
        the autoregressive model (pole model) 
        order of the desired model. 
    * zeros_order: int.
        the moving average model (zeros model) 
        order of the desired model.      
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    
    Returns
    --------------------
    * a: 1d ndarray,
        autoregressive coefficients of the ARMA model.
    * b: 1d ndarray,
        moving average coefficients of the ARMA model.        
    * noise_variace: complex of float,
        variance of model residulas.

    See also
    ------------------
    arma_dubrin,
    arma_prony,
    arma_covar,
    arma_ipz,
    arma_pade. 
    
    Examples:
    
    References
    --------------------
    [1a] M.H. Hayes. Statistical Digital 
        Signal Processing and Modeling, John Wiley & Sons, 1996.
    [1b] https://www.mathworks.com/matlabcentral/fileexchange/2183
                -statistical-digital-signal-processing-and-modeling
    
    '''  
    
    
    x = np.asarray(x)
    N = x.shape[0]
            
    a,_,_ = arma_prony(x, 
                       poles_order, 
                       zeros_order, 
                       mode = 'full')

 
    delta = np.append(1,np.zeros(N-1))
    noise = scipy.signal.lfilter([1],a,delta)
    
    rnoise = operators.lags_matrix(noise, 
                                   lags = zeros_order+1, 
                                   mode = 'prew')
    
#     b = np.dot(np.linalg.pinv(rnoise),x)
    b = scipy.linalg.lstsq(rnoise,x)[0]
    err = np.sum(x*np.conj(x)) -\
            np.dot(np.dot(x.transpose(),rnoise[:,:]),b)

    return a,b,err
   
 