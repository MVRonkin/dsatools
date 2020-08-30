import numpy as np
import scipy

__all__ = ['diff','afft','arg']

def diff(x,delta=1):
    '''
    Difference of samples.
    
    Parameters
    -----------
    * x: 1d ndarray.
    * delta: float,
        value of each step.
    
    Returns
    -----------
    * dx: 1d ndarary.
    
    '''
    x = np.asarray(x)
    dx = (x[2:] - x[:-2])/2
    return np.concatenate(([x[1]-x[0]], dx,[x[-1]-x[-2]] ))/delta

#----------------------------
def afft(x, n_fft=None):
    '''
    
    Amplitude spectrum.
    
    Parameters
    -----------
    * x: 1d ndarray.
    * n_fft: int or None,
        size of fft.
    
    Returns
    -----------
    * dx: 1d ndarary.    
    '''
    
    x = np.asarray(x)
    
    if n_fft is None:
        n_fft = x.size
    
    return np.abs(np.fft.fft(x))

#----------------------------
def arg(x):
    '''
    Unwraped phase.
    
    Parameters
    -----------
    * x: 1d ndarray (complex).
    
    Returns
    -----------
    * phase: 1d ndarary.    
    '''
    
    x = np.asarray(x)

    return np.unwrap(np.angle(x))