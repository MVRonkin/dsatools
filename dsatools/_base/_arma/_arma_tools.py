import numpy as np
import scipy  

from .._polyroot_decomposition import roots2decomposition, roots2freqs

__all__ = ['arma2psd',
           'ar2decomposition',
           'ar2freq',
           'ar2cov',
           'arma2impresponce',
           'ar2predict']

#------------------------------------------------
def arma2psd(a=1, b=1, err=1, n_psd = 512):
    '''
    Compution the power spectral density for a arma model
      in form H(z) = B(z)/A(z), 
      where A and B are polynoms in Z space.
    
    Parameters
    ------------
    * a: 1d ndarray,
        are the autoregression coefficients of the model.
    * a: 1d ndarray,
        are the moving-average coefficients of the model.
    * err: float or complex,
        noise variance of model.
    * n_psd: int,
        number of points in the estimated pseudo-spectrum.
    
    Returns
    --------
    * psd: 1d ndarray,
        pseudo-spectrum in form H(z) = B(z)/A(z).
    
    Notes
    ------
    * If b is 1 it will be oure autoregression model, 
        if a is 1 - moving average model.
    * a and b should be set in full form 
        (a have to including first element (usually 1)).
    
    References
    -----------
    [1] http://thomas-cokelaer.info/
        software/spectrum/html/contents.html
    [2] S.L. Marple, 
        Digital spectral analysis with applications. 
        - New-York: Present-Hall, 1986.
    
    '''
    a = np.append([],a)
    b = np.append([],b)
    numerator   = np.fft.fft(np.append(b, np.zeros(n_psd-b.size)))
    denumerator = np.fft.fft(np.append(a, np.zeros(n_psd-a.size)))
    
    psd = err*numerator*np.conj(numerator)\
        /(denumerator*np.conj(denumerator))
    
#     psd = err*numerator\
#         /(denumerator)    
  
#     psd =psd*np.conj(psd)
 
    return psd.real

#------------------------------------------
def ar2decomposition(x, a, n_psd = None):
    '''
    Autoregression coefficients based signal decomposition.
      The decomposition is based on the polynomial prony method.
    
    Parameters
    ------------
    * x: 1d ndarray,
        signal that will be decomposed.
    * a: 1d ndarray,
        autoregression coefficients, including first 1.
    * n_psd: int or None,
        number of samples in decomposed signal
        if None, n_psd=x.size.
    
    Returns
    ------------
    * psd: 2d darray,
        array with size a.size-1 x x.size.
    
    Notes
    ----------
    * order of output a.size-1.
    
    '''
    x = np.asarray(x)
    N = x.shape[0]

    if (n_psd is None): n_psd = N
    
    order= a.size-1

    
    roots = np.roots(a) #dumps=np.log(np.abs(roots))*fs; freqs = fs*np.angle(roots)/2/np.pi
    
    out = roots2decomposition(x, 
                              roots=roots, 
                              order=order, 
                              n_psd=n_psd)
   
    return out

#----------------------------------------
def ar2freq(a, order = None, fs = 1):
    '''
    Autoregression coefficients based signal decomposition.
      The decomposition is based on the polynomial prony method.
    
    Parameters
    ------------
    * a: 1d ndarray,
        autoregression coefficients, including first 1.
    * order: int or None,
        number of frequencies to estimate,
        if None, order = a.size-1.
    * fs: float or None, sampling frequency.
    
    Returns
    ------------
    * freqs: 1d darray,
        array of frequencies with size = order.
    
    Notes
    ----------
    * order of output a.size-1.
    
    '''

    if (fs is None): fs = N
    
    if order is None:
        order= a.size-1

    
    roots = np.roots(a) #dumps=np.log(np.abs(roots))*fs; freqs = fs*np.angle(roots)/2/np.pi
    
    out = roots2freqs(roots=roots, 
                      order=order, 
                      fs=fs)
   
    return out
#---------------------------------------------   
def ar2predict(x,a,n_predict=1):
    '''
    Linear prediction of new values of signal 
        based on the autoregression model.
    
    Parameters
    -------------
    * x: 1d ndarray,
        input samples for predict.
    * a: 1d ndarray,
        autoregression coefficinets 
        (in full form, with first 1).
    * n_predict: int,
        number of samples to predict.
    
    Returns
    ----------
    * predicted samples:
        (shape 1xn_predict).
    
    References
    ----------
    [1] https://www.mathworks.com/matlabcentral/fileexchange/
        69786-spectral-analysis-linear-prediction-toolbox -
        Spectral Analysis & Linear Prediction Toolbox.
    '''    
    a  = np.asarray(a)
    x  = np.asarray(x)
    N  = x.shape[0]
    
    x_ = np.array(x)
    a_ = -a[1:]  
    p  = a_.size
    for i in range(n_predict):
        x_new = np.inner(a_,x_[-p:][::-1])
        x_ =np.append(x_,x_new)

    return x_[N:]

#--------------------------------------------- 
def arma2impresponce(a,b=1,n_samples=512):
    '''
    Autoregression moving average (ARMA) 
        model to corresponding impulse responce.
    
    Parameters
    ------------
    * a: 1d ndarray,
        autoregression coefficietns.
    * b: 1d ndarray,
        moving average coefficietns.
    * n_samples: int or None,
        number of samples, 
        if None, than n_samples = max(a.size,b.size).
    
    Returns
    ------------
    * h: 1d ndarray,
        impulse responce.
    
    Notes
    ---------
    * If reverce autoregression and moving average
        part - moving average estimation can be obtained.
        
    '''

    a = np.append([],np.asarray(a))
    b = np.append([],np.asarray(b))
    
    if n_samples is None: n_samples = max(a.size,b.size)
    h = np.zeros(n_samples,dtype = a.dtype)
    h[0] = 1.

    return scipy.signal.lfilter(b, a, h)

#------------------------------------------
def ar2cov(a):
    '''
    Estimation of the covariance function
        based on the autoregression coefficients.
    
    Parameters
    -----------
    * a: 1d ndarray,
        autoregression coefficients.
    
    Returns
    -----------
    * cov: 1d ndarray,
        Estimation of covariance function.
    
    References
    ----------
    [1a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [1b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling
    '''
    return _reflection2cov(_ar2reflection(a))
#---------------------------------------------      
def _ar2reflection(a):
    '''
    Reflection coefficients (of latent filter) calculation.
    
    Parameters
    -----------
    * a: 1d ndarray,
        autoregression coefficients.
    
    Returns
    -----------
    * gamma: 1d ndarray,
        reflection coefficients.
    
    References
    ----------
    [1a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [1b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling    
    '''
    a  = np.asarray(a)
    N  = a.shape[0]-1
    a_ = a[1:]/a[0]
    gamma = np.zeros(N,dtype = a.dtype)
    gamma[-1] = a_[-1]

    for j in range(N-1,0,-1):
        a_ = (a_[:j] 
              - gamma[j]*np.conj(a_[:j])[::-1])/(1
                                     - np.abs(gamma[j])**2)
        gamma[j-1] = a_[j-1]
    
    return gamma   
    
#---------------------------------------------    
def _reflection2cov(gamma):
    '''
    Estimation of the covariance function based on the
        input reflection coefficients.
    
    Parameters
    -----------
    * gamma: 1d ndarray,
        reflection coefficients.
    
    Returns
    -----------
    * cov: 1d ndarray,
        Estimation of covariance function.
    
    References
    ----------
    [1a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [1b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling    
    '''
    gamma  = np.asarray(gamma)
    p  = gamma.shape[0]
    aa = np.array([gamma[0]],dtype = gamma.dtype)
    r  = np.array([1,-gamma[0]],dtype = gamma.dtype)

    for j in range(1,p):
        aa=gamma[j]*np.append(np.conj(aa[::-1]),
                              [1])+np.append(aa,[0])
        
        r = np.append(r,-np.inner(r[::-1],aa))
    return r


# #------------------------------------------------
# def ar2decomposition(x, a, n_psd = None):
#     '''
#     Autoregression coefficients based signal decomposition.
#       The decomposition is based on the polynomial prony method.
    
#     Parameters
#     ------------
#     * x: 1d ndarray of decomposed signal.
#     * a: corresponding autoregression coefficients, including first 1.
#     * n_psd: number of samples in decomposed signal.
    
#     Returns
#     ------------
#     * psd: array with size a.size-1 x x.size.
    
#     Notes
#     ----------
#     * order of output a.size-1.
    
#     '''
#     x = np.asarray(x)
#     N = x.shape[0]

#     if (n_psd is None): n_psd = N
    
#     order= a.size-1

    
#     roots = np.roots(a) #dumps=np.log(np.abs(roots))*fs; freqs = fs*np.angle(roots)/2/np.pi
    
#     #TODO: replace on fft to increas stability
# #     v  = matrix.vandermonde(roots, N).T 
    
#     v=np.vander(roots,N=N,increasing=True).T 
  
#     # TODO: look for fast way to solve this equation (based on vandermonde properties)
#     h  = np.dot(np.linalg.pinv(v),x)# amps  = np.abs(h); thets = np.angle(h)    
    
#     out = np.zeros((order-1,N),dtype = x.dtype)
    
#     #TODO: replace on fft to increas stability
#     k = np.arange(N)

#     for i in np.arange(order-1):
#         out[i,:] = h[i]*np.power(roots[i],k) 
   
#     return out