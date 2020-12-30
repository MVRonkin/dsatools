import numpy as np
import scipy


__all__ = ['hilbert','hilbert_autocorrelation']

__EPSILON__ = 1e-8

def hilbert(x):
    '''
    Hilbert transfrom in spetrum domain
      to obtain analitic form of signal.
    
    Parameters
    --------------
    * x: 1d ndarray,
        input signal (real valued or complex valued).
    
    Returns
    ------------
    * sc: 1d ndarray,
        complex samples.
    
    '''
    sp = np.fft.fft(x)
    sp[x.shape[0]//2:] = 0
    return np.fft.ifft(2*sp)


def hilbert_autocorrelation(x):    
    '''
    Hilbert transfrom and autocorrelation function
      both are talen in spetrum domain
      to obtain analitic form of signal correlation
      function.
    
    Parameters
    ------------
    * x: 1d ndarray
        inputs 1d array.
    
    Returns
    ------------
    * x: 1d ndarray,
        complex autocorrelation function.      
                
    Notes
    --------------

    '''
    x = np.asarray(x)
    N = x.shape[0]    
    Sp     = np.fft.fft(x,2*N)
    Sp     = 2*Sp*np.conj(Sp)
    Sp[N:] = 0
    R      = np.fft.ifft(Sp)      
    R      = R[:N]                   
    return  R
