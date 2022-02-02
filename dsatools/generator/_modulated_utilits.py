import numpy as _np

__all__=['calc_phase',
        'linear_frequency',
        'symetric_linear_frequency',
        'poly_frequency',
        'sin_frequency',
        'sin_quart_frequency',
        'sin_half_frequency',
        'sin_phase_frequency'
        ]

#-------------------------------
def calc_phase(frequencies, init_phases, length = None, fs=None):
    '''
    Utilite to calculte the phase to time relation
      based on frequency to time relation.
      ..math::
      phi(n) = 2pi*integral(*f(n)*n/fs)
    
    Paramters
    -----------
    frequencies: float or 1d ndarray,
      could be either carier frequency (float),
      or frequency to time relation (frequency modulation).  
    init_phases: float or 1d ndarray,  
      could be either initial phase (float),
      or additional phase to time relation (phase modulation).
    length: int or None,
      length or required signal, 
      if None, length = max of lengths 
      between frequencies, amplitudes, init_phases
    fs: float or None,
      sampling frequency, if None, fs = length
    
    Returns
    --------------
    full_phase: 1d ndarray.
    
    Note
    ----------
    If some of the parameters (freq, amp, phase) have a size,
      smaller than the required size, it will be extended,
      with the last value. In the opposite case, 
      it will be cut up to the required size.
    
    '''
    frequencies = _np.asarray(frequencies, dtype = float)
    init_phases = _np.asarray(init_phases, dtype = float)
    
    if length is None:
        length = max(frequencies.size,
                     init_phases.size)
 
    frequencies = _check_size(frequencies, length)
    init_phases = _check_size(init_phases, length) 
    
    if fs is None:
        fs = length
        
    dt = 1/fs
    
    return 2*_np.pi*_np.cumsum(frequencies)*dt+init_phases


#---------------------------------------
def to_beatsignal(sig_ref, sig_delayed,
                  reduced_ratio=None, hilbert = False):
    '''
    Mixing reference and delayed (received)
        signals to create the beat one.
    
    Parameters
    -------------
    sig_ref: 1d ndarray,
        reference signal
    sig_delayed: 1d ndarray,
        dealyed signal
    reduced_ratio: float,
        pooling (or downsampling) ratio as fs_old/fs_new,
        where fs is the sampling frequency,
    hilbert: bool,
        If ture, hilbert transform will be carried out additionally.
    
    Returns
    -------------
    sig_beat: 1d ndarray,
        beat signal.
    '''
    sig_beat = sig_ref*_np.conj(sig_delayed);
    if reduced_ratio is not None:
        
        len_reduce = int(len(sig_ref)/reduced_ratio)
        
        sp = _np.fft.fft(sig_beat)
        
        if hilbert is False:
            sp = _np.concatenate((sp[:len_reduce//2],
                                 [sp[len(sp)//2]],
                                 sp[len(sp) - len_reduce//2 +1:]))
        else:
            sp = _np.concatenate((sp[:len_reduce//2],
                                 [sp[len(sp)//2]],
                                 _np.zeros(len_reduce//2 +1,
                                          dtype=complex)))
            
        sig_beat = _np.fft.ifft(sp)
    else:
        if hilbert:
            sp = _np.fft.fft(sig_beat)
            sp[len(sp)//2:] = 0
            sig_beat = _np.fft.ifft(sp)
 
    return sig_beat

#-------------------------------
def linear_frequency(f0, delta_f, length, fs):
    '''
    Linear frequency to time relation.
      ..math::
      f(n) = f0 + delta_f*(n/fs)/Tm,
      Tm = N/fs, n=0,...,N-1

    Paramteres
    ------------
    f0: float,
      carried frequency.
    delta_f: float,
      frequency bandwidth.
    length: int,
      length of relation.
    fs: float,
      sampling frequency.
    
    Return
    ----------
    freq: 1d ndarray.
    '''
    length = int(length)
    t = _np.arange(length)/fs
    return f0+delta_f*t/((length/fs))

#-------------------------------
def symetric_linear_frequency(f0, delta_f, length, fs):
    '''
    Linear frequency to time relation.
      ..math::
      f(n) = [f0 + delta_f*(n/fs)/Tm,
      f0+delta_f - delta_f*(n/fs)/Tm]
      Tm = N/2fs, n=0,...,N-1

    Paramteres
    ------------
    f0: float,
      carried frequency.
    delta_f: float,
      frequency bandwidth.
    length: int,
      length of relation.
    fs: float,
      sampling frequency.
    
    Return
    ----------
    freq: 1d ndarray.
    '''
    length = int(length)
    t = _np.arange(length)/(2*fs)
    tm = length/fs/2
    return _np.concatenate((f0+delta_f*t/tm, 
                           f0+delta_f-delta_f*t/tm))
#-------------------------------
def poly_frequency(f0, delta_f, degree, length, fs):
    '''
    Polynomial frequency to time relation.
      ..math::      
      f(n) = f0 + delta_f*(fm/max(fm)),
      fm = (n/fs)^d/Tm
      Tm = N/fs, n=0,...,N-1

    Paramteres
    ------------
    f0: float,
      carried frequency.
    delta_f: float,
      frequency bandwidth.
    degree: float,
      degree of Polynomial relation.
    length: int,
      length of relation.
    fs: float,
      sampling frequency.
    
    Return
    ----------
    freq: 1d ndarray.
    '''
    t = _np.arange(length)/fs
    freq = _np.power(t,degree)/((length/fs))    
    return f0+delta_f*freq/_np.max(freq)
#-------------------------------
def sin_frequency(f0, delta_f, length, fs):
    '''
    Sinus frequency to time relation.
      ..math::      
      f(n) = f0+delta_f/2+delta_f*sin(2pi*n/(Tmfs))/2,
      Tm = N/fs, n=0,...,N-1

    Paramteres
    ------------
    f0: float,
      carried frequency.
    delta_f: float,
      frequency bandwidth.
    length: int,
      length of relation.
    fs: float,
      sampling frequency.
    
    Return
    ----------
    freq: 1d ndarray.
    '''
    t = _np.arange(length)/fs
    tm = length/fs
    return (f0+delta_f/2)+delta_f*_np.sin(2*_np.pi*t/tm)/2
#-------------------------------
def sin_quart_frequency(f0, delta_f, length, fs):
    '''
    Sinus quarter (first quadrant) frequency to time relation.
      ..math::      
      f(n) = f0+delta_f*sin(2pi*n/(4Tmfs)),
      Tm = N/fs, n=0,...,N-1

    Paramteres
    ------------
    f0: float,
      carried frequency.
    delta_f: float,
      frequency bandwidth.
    length: int,
      length of relation.
    fs: float,
      sampling frequency.
    
    Return
    ----------
    freq: 1d ndarray.
    '''
    t = _np.arange(length)/fs
    tm = length/fs
    return (f0)+delta_f*_np.sin((2*_np.pi*t/tm)/4)

#-------------------------------
def sin_half_frequency(f0, delta_f, length, fs):
    '''
    Sinus half (first 2 quadrants) frequency to time relation.
      ..math::      
      f(n) = f0+delta_f*sin(2pi*n/(2Tmfs)),
      Tm = N/fs, n=0,...,N-1

    Paramteres
    ------------
    f0: float,
      carried frequency.
    delta_f: float,
      frequency bandwidth.
    length: int,
      length of relation.
    fs: float,
      sampling frequency.
    
    Return
    ----------
    freq: 1d ndarray.
    '''
    t = _np.arange(length)/fs
    tm = length/fs
    return (f0)+delta_f*_np.sin((2*_np.pi*t/tm)/2)

#-------------------------------
def sin_phase_frequency(f0, delta_f, length, fs, phi0=0, phiN=_np.pi/4):
    '''
    Arbitarry sinus frequency to time relation.
      ..math::      
      f(n) = f0+delta_f*sin(2pi*n/(Tm*fs*pi/phiN)+phi0),
      Tm = N/fs, n=0,...,N-1

    Paramteres
    ------------
    f0: float,
      carried frequency.
    delta_f: float,
      frequency bandwidth.
    length: int,
      length of relation.
    fs: float,
      sampling frequency.
    
    Return
    ----------
    freq: 1d ndarray.
    '''
    t = _np.arange(length)/fs
    tm = length/fs
    divider = _np.pi/phiN
    freqm = _np.sin(2*_np.pi*t/(tm*divider)+phi0)
    return (f0)+delta_f*(freqm-min(freqm))
            /(max(freqm)-min(freqm))

#-------------------------------------
def exp_decay_frequency(f0, delta_f, decay, length, fs): 
    '''
    Exponential Decay frequency to time relation.
      ..math::
      f(n) = f0 + delta_f*exp(-2*pi*alpha*n/N),
      n=0,...,N-1

    Paramteres
    ------------
    f0: float,
      carried frequency.
    delta_f: float,
      frequency bandwidth.
    decay: float,
      coefficient on exp decay.  
    length: int,
      length of relation.
    fs: float,
      sampling frequency.
    
    Return
    ----------
    freq: 1d ndarray.
    '''
    t = np.arange(length)/fs
    tm = length/fs
    return (f0)+delta_f*np.exp(-2*decay*np.pi*t/tm)

#-------------------------------------
def _check_size(array, size=None):
    
    array = _np.asarray(array, dtype = float)
    
    if size is None:
        return array
    
    size = int(size)
    
    if array.size==1:
        return _np.ones(size, dtype=float)*array 
    else:
        if array.size>=size:
            return array[:size]
        
        else:
            last  = _np.ones(size-array.size, 
                            dtype=float)*array[-1]
            return _np.concatenate((array,last))
