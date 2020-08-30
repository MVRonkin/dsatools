import random
import numpy as np
import struct
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy  

__all__ = ['ma_yule_walker']

from ... import utilits as ut

from ._ar_yule_walker import ar_yule_walker
#------------------------------------------------------------------
def ma_yule_walker(x, poles_order, zeros_order, mode = 'full', unbias = False):
    '''
    Modified Yule-Walker method for 
        Moving Average(MA) model parameters estimation.

    Parameters
    --------------
    * x: 1d ndarray,
        1-d input ndarray.    
    * poles_order: int,
        is the order of the 
        firstly estiamted autoregressive model.
    * zeros_order: int,
        is the order of the 
        desired moving-average model.        
    * mode: string,
        mode of correlation function, 
        mode = {full, same, straight}.
    * unbias: bool, 
        if True, unbiased covariance 
        function will be taken.
    
      
    Returns
    ------------------
    * a: 1d ndarray (complex (or float)),
        autoregression coefficients of the auxilary,
        fristly estimated autoregressive model. 
    * b: 1d ndarray (complex (or float)),
        moving-average coefficients of 
        the desired moving-average model.        
    * noise_variace: complex (or float), 
        variance of model residulas.
              
    Notes
    ---------
    
    References
    --------------
    [1a] P. Stoica, R.L. Moses, 
        Spectral analysis of signals 
        - New-York: Present-Hall, 2005.
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page.
    [2]  S.L. Marple, 
        Digital spectral analysis with applications. 
        – New-York: Present-Hall, 1986.
    
    '''
    x = np.asarray(x)
    N = x.shape[0]
    
    #TODO: check orders
    a, var = ar_yule_walker(x, 
                            order  = poles_order, 
                            mode   = mode, 
                            unbias = unbias)
    
    b,_    = ar_yule_walker(x, 
                            order  = zeros_order, 
                            mode   = mode, 
                            unbias = unbias)   
    b = b[1:]

    return a,b, var


    
   