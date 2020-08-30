import numpy as np
import scipy  


from ... import operators

from ._ar_ls import ar_ls
from ._ma_yule_walker import ma_yule_walker
from ._ma_dubrin import ma_dubrin
from ._ma_innovations import ma_innovations

__all__ = ['arma_covar']

#------------------------------------------------------------------
def arma_covar(x, poles_order, zeros_order=0):
    '''
    Autoregression-Moving Average (ARMA) model based on the 
        2 stages covariance autoregression model
        and Yule-Walker moving average model.  
    
    Parameters
    ----------------------
    * x: 1d ndarray.
    * poles_order: int,
        is the orders of poles 
            (denumenator) polynom of the ARMA model.
    * zeros_order: int, 
        is the orders of zeros (numenator) polynom 
        of the ARMA model, if zeros_order =0, than AR model 
        of covariance matrix will be returned.
    
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
    arma_pade. 
    
    Examples:
    
    References
    --------------------
    [1a] P. Stoica, R.L. Moses, Spectral analysis of signals 
                            - New-York: Present-Hall, 2005.
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page
    [2a] M.H. Hayes. Statistical Digital 
        Signal Processing and Modeling, John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/2183
                -statistical-digital-signal-processing-and-modeling
    [3]  S.L. Marple, Digital spectral analysis with applications.
                        – New-York: Present-Hall, 1986.
    
    '''
    x = np.asarray(x)
    N = x.shape[0]

    r = operators.correlation(x, 
                              mode='same', 
                              unbias=False)    
    a,var = ar_ls(r, order=poles_order,mode='covar')
    
    if(zeros_order>0):
        noises = scipy.signal.lfilter([1],a[1:],x)[poles_order:]

        _,b, var = ma_yule_walker(noises, 
                                  zeros_order, 
                                  2*zeros_order,
                                  mode='same',
                                  unbias=False)     

    return a,b,var    
    

#------------------------------------------------------------------    
def arma_dubrin(x, poles_order, zeros_order=0):
    '''
    Autoregression-Moving Average (ARMA) model based on the 
        2 stages covariance autoregression model
        and Dubrin moving average model.  
    
    Parameters
    ----------------
    * x: 1d ndarray.
    * poles_order: int,
        is the orders of poles 
        (denumenator) polynom of the ARMA model.
    * zeros_order: int, 
        is the orders of zeros 
        (numenator) polynom of the ARMA model,if zeros_order =0, 
        than AR model of covariance matrix will be returned.
    

    Returns
    --------------
    * a: 1d ndarray,
        autoregressive coefficients of the ARMA model.
    * b: 1d ndarray,
        moving average coefficients of the ARMA model.        
    * noise_variace: complex of float,
        variance of model residulas.
    
    Notes
    ---------------
    
    See also 
    ------------------
    arma_covar,
    arma_prony,
    arma_shanks,
    arma_ipz,
    arma_pade. 
    
    Examples
    -------------
    
    References
    --------------------
    [1a] P. Stoica, R.L. Moses, Spectral analysis of signals 
                            - New-York: Present-Hall, 2005.
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page
    [2a] M.H. Hayes. Statistical Digital 
        Signal Processing and Modeling, John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/2183
                -statistical-digital-signal-processing-and-modeling
    [3]  S.L. Marple, Digital spectral analysis with applications.
                        – New-York: Present-Hall, 1986.
    
    '''
    x = np.asarray(x)
    N = x.shape[0]
   
    r = x#signals.correlation.correlation(x,mode='same', unbias=True)[N//2:]
    
    a,var = ar_ls(r, order=poles_order,mode='full')
    
    if(zeros_order>0):
        noises = scipy.signal.lfilter([1],a,x)#[poles_order:]

        _,b, var = ma_dubrin(noises, zeros_order, 2*zeros_order)     
    
    else:
        b = 1

    return a,b,var

#------------------------------------------------------------------     
def arma_innovations(x, poles_order, zeros_order):
    '''
    Autoregression-Moving Average (ARMA) model based on the 
        2 stages covariance autoregression model
        and innovation moving average model.  
    
    Parameters
    ----------------
    * x: 1d ndarray.
    * poles_order: int,
        is the orders of poles 
        (denumenator) polynom of the ARMA model.
    * zeros_order: int,
        is the orders of zeros 
        (numenator) polynom of the ARMA model,if zeros_order =0, 
        than AR model of covariance matrix will be returned.

    Returns
    ---------------
    * a: 1d ndarray,
        autoregressive coefficients of the ARMA model.
    * b: 1d ndarray,
        moving average coefficients of the ARMA model.        
    * noise_variace: complex of float,
        variance of model residulas.
    
    Notes
    ---------------
    
    See also 
    --------------
    arma_covar,
    arma_dubrin,
    arma_prony,
    arma_shanks,
    arma_ipz,
    arma_pade. 
    
    Examples
    -------------
    
    References
    -------------
    [1] Brockwell, P. J., & Davis, R. A. 
        Time series: Theory and methods (2nd ed.).
        New York: Springer, 1991.
    [2a] P. Stoica, R.L. Moses, Spectral analysis of signals 
                            - New-York: Present-Hall, 2005.
    [2b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page
    [3]  S.L. Marple, Digital spectral analysis with applications.
                        – New-York: Present-Hall, 1986.
    '''
    x = np.asarray(x)
    N = x.shape[0]

    r=x
    r = operators.correlation(x,mode='full', unbias=False)    
#     a,var = burg(r, order=poles_order)
    
    a,var=ar_ls(r,poles_order,mode='covar')
    
    if(zeros_order>0):
        
        noises = scipy.signal.lfilter([1],-a[1:],x)#[poles_order:]

        _,b, var = ma_innovations(noises, zeros_order)     
    
    else:
        b = 1

    return a,b,var
    