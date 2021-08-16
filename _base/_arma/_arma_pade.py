import numpy as np
import scipy  

from ... import operators
from ... import utilits as ut

__all__ = ['arma_pade']
#------------------------------------------------------------------
def arma_pade(x, poles_order, zeros_order, mode = 'full'):
    '''
    Autoregression-Moving Average (ARMA) model based on the 
            and Pade signal model.  
    
    Parameters
    ---------------------
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
    
    Notes
    --------------
   
    
    See also
     ------------------
    arma_dubrin,
    arma_prony,
    arma_shanks,
    arma_ipz,
    arma_covar. 
    
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

    r  = operators.lags_matrix(x,
                               lags = poles_order+1, 
                               mode = mode)
    rn = r[zeros_order+1:zeros_order+poles_order,1:]
    r1 = r[zeros_order+1:zeros_order+poles_order,0]
    
#     a  = np.dot(np.linalg.pinv(-rn),r1)
    a = scipy.linalg.lstsq(-rn,r1)[0]
    a  = np.append(1,a)
    
    rn = r[:zeros_order+1,:poles_order+1]
    b  = np.dot(rn,a)
   
    err = 1

    return a,b,err



