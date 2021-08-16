import numpy as np
import scipy

from ... import operators

__all__ = ['kernel_periodogram']
#---------------------------------------------
def kernel_periodogram(x, mode='full', kernel='linear', kpar=1, 
                            window= None, lags = None, n_psd = None):
    '''
    Kernel basded periodogram-windowed spetrum estimation.
     
    Parameters 
    ------------
    * x: 1d ndarray.    
    * mode: string, 
        covariance matrix mode,
        mode = {full,covar,traj,toeplitz}.
    * kernel: string,
         kernel type (linear for defoult similar as
         traditional covariance function), 
         kernals = {linear, rbf, thin_plate, sigmoid, poly, 
                    bump, polymorph, euclid}.
    * kpar: float,
        kernel parameter.
    * window: string or tuple (string, value),
            window type (square window if None).
    * lags: int or None,
        number of lags in kernel (x.shape[0]//2 if None).
    * n_psd: int or None,  
        Length of psceudo-spectrum (Npsd = x.shape[0] if None).
    
    Returns
    -----------------------
    * periodogram:  1d ndarray.

    Notes
    ------------
    Scipy Window types:
        - `~scipy.signal.windows.boxcar`
        - `~scipy.signal.windows.triang`
        - `~scipy.signal.windows.blackman`
        - `~scipy.signal.windows.hamming`
        - `~scipy.signal.windows.hann`
        - `~scipy.signal.windows.bartlett`
        - `~scipy.signal.windows.flattop`
        - `~scipy.signal.windows.parzen`
        - `~scipy.signal.windows.bohman`
        - `~scipy.signal.windows.blackmanharris`
        - `~scipy.signal.windows.nuttall`
        - `~scipy.signal.windows.barthann`
        - `~scipy.signal.windows.kaiser` (needs beta)
        - `~scipy.signal.windows.gaussian` (needs standard deviation)
        - `~scipy.signal.windows.general_gaussian` (needs power, width)
        - `~scipy.signal.windows.slepian` (needs width)
        - `~scipy.signal.windows.dpss` (needs normalized half-bandwidth)
        - `~scipy.signal.windows.chebwin` (needs attenuation)
        - `~scipy.signal.windows.exponential` (needs decay scale)
        - `~scipy.signal.windows.tukey` (needs taper fraction)
    
            
    References
    --------------------
    
    Example
    ---------------------- 
    
    See also
    ----------------------
    correlogram
    periodogram
    bartlett
    welch
    blackman_tukey
    daniell
    
    '''
    x = np.asarray(x)
    N = x.shape[0]
    
    if lags is None: lags = N//2
    if n_psd is None: n_psd = N
        
    R = operators.kernel_matrix(x,
                                mode=mode,
                                kernel=kernel,
                                kpar=kpar,
                                lags=lags+1,
                                ret_base=False,)
    
    
    rt = R[0,1:lags+1] 

    r = np.concatenate((np.conj(rt[::-1]),[R[0,0]], rt[:-1] ))

    
    if window is not None:
        M = r.size
        w = scipy.signal.get_window(window,M)    
        r  = r*w
    
 
    px = np.abs(np.fft.fft(np.append(r,[0]),n_psd))
    px[0]=px[1]
    
    return px
