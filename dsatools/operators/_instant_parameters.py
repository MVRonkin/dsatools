import numpy as np
import scipy.signal
__all__ = ['instant_parameters']
#-----------------------------------
def instant_parameters(signal, fs = None):
    '''
    Instant parameters estimation:
    ..math::
        analitc_signal = hilbert(signal)
        envelope  = |analitc_signal|
        phase     = unwrap(angle(analitc_signal))
        frequency = diff(phase)
    
    Paramteres
    -------------
    signal: 1d ndarray,
        input signal (can be real or complex);
    fs: float or None,
        sampling frequecny, fs = signal.size, if None
    
    Return
    -------------
    frequency: 1d ndarray,
        instant frequency to time relation.
    envelope: 1d ndarray,
        instant envelope to time relation. 
    phase: 1d ndarray,
        instant pahse to time relation.
    
    '''
    if fs is None:
        fs = len(signal)
    
    signal = np.asarray(signal)
    
    if signal.dtype != complex:
        analytic = scipy.signal.hilbert(signal)
    else:
        analytic = signal

    envelope = np.abs(analytic)
    angles   = np.angle(analytic)
    phase    = np.unwrap(angles)

    frequency = np.concatenate((np.diff(angles),
                                [angles[-2] - angles[-1]]))

    for i in range(frequency.size):
        if frequency[i]< 0:
            if i>0: frequency[i] = frequency[i-1]
            else:   frequency[i] = frequency[i+1]  


    frequency = frequency*fs/(2*np.pi)
    
    return frequency, envelope, phase
