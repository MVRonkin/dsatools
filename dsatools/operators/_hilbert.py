import numpy as np
import scipy


__all__ = ['hilbert']

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
