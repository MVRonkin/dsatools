import numpy as np
import scipy  

from ... import utilits as ut

from ._ar_ls import ar_ls 

__all__ = ['arma_ipz']
#------------------------------------------------------------------
def arma_ipz(x, poles_order, zeros_order=0, n_iter=1):
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
    * n_iter: int,
        number ofitteration to estimate ARMA coefficients.
    
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
    arma_covar,
    arma_pade, 
    arma_hannan_rissanen.
    
    Examples
    --------------------
    
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
    
    '''
    x = np.asarray(x)
    N = x.shape[0]
    
    a,_ = ar_ls(x, poles_order, mode='full')
#     a,_,_ = signals.spectrum.arma_prony(x, poles_order,zeros_order, n_psd=-1)
    
    for i in np.arange(n_iter):
        noise = scipy.signal.lfilter(a[:],[1],x)
        L = poles_order+zeros_order

        Z1 = (scipy.linalg.toeplitz(x)[L:-1,:poles_order])
        Z2 = (scipy.linalg.toeplitz(noise)[L:-1,:zeros_order])

        Z  = np.append(Z1,-Z2,axis=1)

        z = x[L+1:]
#         ab = np.dot(np.linalg.pinv(-Z),z)
        ab = scipy.linalg.lstsq(-Z,z)[0]
        a = np.append([1],ab[:poles_order])
        b = np.append([1],ab[poles_order:])
#   err = np.linalg.norm(np.dot(Z,ab) + z)/(N-L) 
    err = 1
    return a,b,err
    
#
