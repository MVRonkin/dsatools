import numpy as np

from . _modulated_utilits import _check_size
__all__ = ['modulated_signal']

#------------------------------------------------------------------------------------
def modulated_signal(frequencies, 
                     amplitudes, 
                     init_phases, 
                     length=None, 
                     fs=None, 
                     delay = 0, 
                     ret_complex = True):
    '''
    Build one-mode arbitarry modulated signal.
    s(n) = a(n)exp(1j[phi(n)+phi0(n)]),
        phi(n) = 2pi*integral(*f(n)*n/fs)
    
    Paramters
    -----------
    frequencies: float or 1d ndarray,
      could be either carier frequency (float),
      or frequency to time relation (frequency modulation).  
    amplitudes: float or 1d ndarray,  
      could be either amplitude level (float),
      or amplitude to time relation (amplitude modulation).
    init_phases: float or 1d ndarray,  
      could be either initial phase (float),
      or additional phase to time relation (phase modulation).
    length: int or None,
      length or required signal, 
      if None, length = max of lengths 
      between frequencies, amplitudes, init_phases
    fs: float or None,
      sampling frequency, if None, fs = length
    delay: float,
      if delay>0, then add shift in signal start: int(delay*fs),
      and additonal initail phase shift:
        2*np.pi*ddelay*frequencies[0], 
        ddelay = (delay*fs-int(delay*fs))/fs,  
    ret_complex: bool,
      if true - complex valued signal will be returned.
    
    Returns
    --------------
    signal: 1d ndarray.
    
    Note
    ----------
    The Result is similar to Intrinsic-mode-function.
    If some of the parameters (freq, amp, phase) have a size,
      smaller than the required size, it will be extended,
      with the last value. In the opposite case, 
      it will be cut up to the required size.
    This function work with only one signal, 
      if you want superposition of signals, 
      make sum of severeal function calls.

    Examples
    ----------------
    import numpy as np
    import dsatools.utilits as ut
    from dsatools.generator import modulated_signal
    N = 512
    FS = 20
    f0 = 1
    delta_f = 0.3
    freq = f0+delta_f*np.arange(N)/N 
    s = modulated_signal(freq, 
                         amplitudes = 1, 
                         init_phases = 0, 
                         length=N, 
                         fs=FS, 
                         delay = 13.043/FS, 
                         ret_complex = True)
    ut.probe(s)
    
    '''
    frequencies = np.asarray(frequencies, dtype = float)
    amplitudes  = np.asarray(amplitudes,  dtype = float)
    init_phases = np.asarray(init_phases, dtype = float)
    
    if length is None:
        length = max(amplitudes.size,
                     frequencies.size,
                     init_phases.size)
        
    if fs is None:
        fs = length
 
    dt = 1/fs    
 
    frequencies = _check_size(frequencies, length)
    amplitudes  = _check_size(amplitudes,  length)
    init_phases = _check_size(init_phases, length) 
    
    full_phase = 2*np.pi*np.cumsum(frequencies)*dt+init_phases
    
    if delay>0:
        n_delay = int(delay*fs)
        ddelay  = (delay*fs-n_delay)/fs
        dphi0   =  2*np.pi*ddelay*frequencies[0]
        amplitudes = np.concatenate((np.zeros(n_delay),
                                     amplitudes[:-n_delay]))
        full_phase = np.concatenate((np.zeros(n_delay),
                                     full_phase[:-n_delay]+dphi0))        
    
    if(ret_complex):
        return amplitudes*np.exp(1j*full_phase)
    else:
        return amplitudes*np.cos(full_phase)
