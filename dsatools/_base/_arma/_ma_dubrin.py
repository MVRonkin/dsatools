import random
import numpy as np
import struct
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy  

__all__ = ['ma_dubrin']

from ... import utilits as ut

from ._ar_ls import ar_ls
#------------------------------------------------------------------

def ma_dubrin(x, poles_order, zeros_order,mode='full'):
    '''
    Moving Average (MA) model based on the Dubrin method.  
    
    Parameters
    -------------
    * x: 1d ndarray,
        input signal.
    * poles_order: int,
        is the orders of auxilary autoregressive model (denumenator).
    * zeros_order: int,
        is the orders of zeros (numenator) polynom of the MA model.
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
        
    Returns
    ----------
    * a: complex 1d ndarray,
        coefficients of the auxilary, firstly appoximated 
        autoregressive model.
    * b: complex 1d ndarray,
        are the coefficients of the desired moving-average model.
    * noise_variace complex, 
        variance of model residulas.

    Notes
    ---------
    
    See also
    ----------
    ma_innovations,
    ma_yule_walker
    
    Examples
    -------------
    
    References
    ------------
    [1a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [1b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling
        
    '''      
    x = np.asarray(x)    
    N = x.shape[0]  

    a,err = ar_ls(x,poles_order,mode=mode)
    
    a = N*a/np.sqrt(err)
    
    b,err = ar_ls(a,zeros_order,mode=mode)

    return a,b, err



