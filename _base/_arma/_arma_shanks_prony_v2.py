import random
import numpy as np
import struct
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy  



from ... import matrix
from ... import utilits as ut

f

__all__ = ['arma_shanks_v2','arma_prony_v2']
#-----------------------------------------------------------
def arma_prony_v2(x, poles_order, zeros_order, mode = 'full', n_psd = None):
    '''
    Autoregression-Moving Average (ARMA) model based on the 
            based on the Prony least-sqaure.   
    
    Parameters
    ----------------------
    * x: 1d ndarray.
    * poles_order: is the orders of poles 
            (denumenator) polynom of the ARMA model.
    * zeros_order: is the orders of zeros (numenator) polynom 
        of the ARMA model, if zeros_order =0, than AR model 
                        of covariance matrix will be returned.
    * n_psd: length of desired pseudospctrum 
              (if None, n_psd = x.shape[0]), if n_psd<0, 
              then model coefficients poles_order and zeros_order
                and noise_variance (\sigma^2) will be returend. 
    
    Returns
    --------------------
    > if n_psd>0: 
          * pseudo-spectrum,
    > else: 
       * a,b: are the coefficients of the ARMA model.
       * noise_variace - variance of model residulas.
    
    Notes
    --------------
    The ARMA model concidered in the following form:
    ..math::
    H(z) = B(z)/A(z), 
    where:
    * H(z) is the estimated function.
    * B(z) is the numerator polynom of order zeros_order;    
    * A(z) is the denumerator polynom of poles_order.
    
    See also
    ------------------
    arma_dubrin,
    arma_covar,
    arma_shanks,
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
    r = matrix.lags_matrix(x,mcolumns = poles_order+1, mode = mode)  
    if(n_psd is None): n_psd = N
    
    r1 = r[zeros_order+1:,0]
    rn = r[zeros_order:-1,:poles_order]
#     print(r1.shape,rn.shape)
#     a  = np.dot(np.linalg.pinv(-rn),r1) 
    a = scipy.linalg.lstsq(-rn,r1)[0]
    a  = np.append(1, a)   
    
    rn = r[:zeros_order+1,:]
    b  = np.dot(rn, a)
    
    err = 1
    C = np.dot( r1.transpose(), r[zeros_order+1:]),
    err=np.dot(C,a)
    
    if(n_psd<1):
        return a,b,err
    else:
        psd = ut.arma2psd(a,b,np.abs(err),n_psd)
        return psd
    
def arma_shanks_v2(x, poles_order, zeros_order, mode = 'full', n_psd = None):
    '''
    Autoregression-Moving Average (ARMA) model based on the 
            Shanks signal model.  
    
    Parameters
    ----------------------
    * x: 1d ndarray.
    * poles_order: is the orders of poles 
            (denumenator) polynom of the ARMA model.
    * zeros_order: is the orders of zeros (numenator) polynom 
        of the ARMA model, if zeros_order =0, than AR model 
                        of covariance matrix will be returned.
    * n_psd: length of desired pseudospctrum 
              (if None, n_psd = x.shape[0]), if n_psd<0, 
              then model coefficients poles_order and zeros_order
                and noise_variance (\sigma^2) will be returend. 
    
    Returns
    --------------------
    > if n_psd>0: 
          * pseudo-spectrum,
    > else: 
       * a,b: are the coefficients of the ARMA model.
       * noise_variace - variance of model residulas.
    
    Notes
    --------------
    The ARMA model concidered in the following form:
    ..math::
    H(z) = B(z)/A(z), 
    where:
    * H(z) is the estimated function.
    * B(z) is the numerator polynom of order zeros_order;    
    * A(z) is the denumerator polynom of poles_order.
    
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
    
    if n_psd is None: n_psd = N
    if mode == 'full': mode = 'prew'
        
    a,_,_ = arma_prony_v2(x, poles_order, 
                       zeros_order=0, mode = mode, n_psd = -1)

    delta = np.append(1,np.zeros(N-1))
    noise = scipy.signal.lfilter([1],a,delta)
    
    rnoise = matrix.lags_matrix(noise, 
                 mcolumns = zeros_order+1, mode = mode)
    
#     b = np.dot(np.linalg.pinv(rnoise),x[:rnoise.shape[0]])
    b = scipy.linalg.lstsq(rnoise,x[:rnoise.shape[0]])[0]
    
    err =np.sum(x[:rnoise.shape[0]]*np.conj(x[:rnoise.shape[0]])) -\
            np.dot(np.dot(x[:rnoise.shape[0]].transpose(),rnoise[:,:]),b)
    
    if(n_psd<1):
        return a,b,err
    else:
        return ut.arma2psd(a,b,np.abs(err),n_psd) 