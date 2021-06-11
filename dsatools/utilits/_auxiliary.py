import numpy as np
from . _awgn import wgn_with_snr, signal_like_noise

_all__ = ['pad_to_power_of_2','pad_noises']
#---------------------------------------    
def pad_to_power_of_2(signal,
                      method = None,
                      param = 0,
                      degree = None,
                      random_state = None):
    '''
    Padding to length as power of 2.
    
    Paramteres
    -----------------
    * signal: 1d ndarray,
        input signal.
    * method, string,
        method of padding:
        * None or 'constant'
            - with param value;
        * 'zeroing' - with 0;    
        * 'reflect', with the the mirrored 
                        samples, without last.
        * 'symmetric', with the the mirrored 
                        samples, with last.
        * 'cyclic', with repeat from first sample.
        * 'linear_ramp' with samples from 
                last value to 0
        * 'wgn': with noise of power param;
        * 'wng_db': White Gaussian Noises
              with param = SNR in dB  ; 
        * 'signal_noise': with noise as
               random signal samples with 
               param = SNR in dB ;
    * param: float,
        auxiliary parameter.
    * degree: int or None,
        if None, than size will be padded to
           the nearest ceil degree of 2;
        if 2^degree<=signal size, than
             only first 2^degree will be returned
        if 2^degree>signal size, than
              padding will be perfermed.
    * random_state: float or None,
        random_state seed for noises.
        
    Returns
    ------------
    * padded signal.    
    '''       
    N = len(signal)
    
    if degree is None:
        N_new = np.power(2,int(np.log2(N))+1)
    elif np.power(2,degree)<=N:
        return signal[:np.power(2,degree)]
    else:
        N_new = np.power(2,degree)

    if method is 'zeroing':
        param = 0
        method = 'constant'
    
    if method is None or method is 'constant':
        return np.pad(signal,
                      (0,N_new-N),
                      'constant', 
                      constant_values = param)
    
    elif method in ['reflect','symmetric','linear_ramp']:
        return np.pad(signal,(0,N_new-N),method)
     
    elif method in ['wng_db','wgn','signal_like']:
        return pad_noises(signal, 
                          pad_width = [0,N_new-N], 
                           snr = param, 
                           method = method, 
                           random_state = random_state)
    
    elif method is 'cyclic':
        return np.tile(signal,N_new//N+1)[:N_new]

    else:
        return ValueError('uncorrect value')

#---------------------------------------    
def pad_noises(signal, 
               pad_width = [0,0], 
               snr = 30, 
               method = 'wng_db', 
               random_state = None):
    '''
    Padding with pad_width.
    
    Paramteres
    -----------------
    * signal: 1d ndarray,
        input signal.
    * pad_width: int list, [before, after],
        number of values add before and after,
        if only one value - then only after.
    * snr: float, or None,
        Signal-to-Noise ratio in dB by default,
        could have different meaning depends 
        on the method.
        If None, than zeros will be padded.    
    * method: string,
        * 'wgn_db': padding with 
            White Gaussian Noises, SNR in dB.
        * 'wgn': padding with 
            Noise Power, in abs value. 
        * 'signal_like': padding with 
            Noises taken as random signal samples,
            with SNR in dB.             
    * random_state: float,
        random state.
     
    Returns
    ------------
    * padded signal.
    
    ''' 
    signal = np.asarray(signal)
    pad_width = np.asarray(pad_width,dtype=int)
    if pad_width.size <2:
        pad_width = np.array([0, 
                              np.squeeze(pad_width)])
    if snr is None:
        return np.pad(signal,pad_width)
    
    if method is 'wgn_db':
        return np.concatenate((
                       wgn_with_snr(signal,snr = snr, 
                                    length=pad_width[0],
                                    random_state = random_state),
                       signal, 
                       wgn_with_snr(signal,snr = snr, 
                                    length=pad_width[1],
                                    random_state = random_state),
                              ))
    elif method is 'wgn':
        return np.concatenate((
                       wgn(snr, 
                              length=pad_width[0],
                              is_complex = (
                                  signal.dtype == complex),
                               random_state = random_state),
                       signal, 
                       wgn(snr, 
                              length=pad_width[1],
                              is_complex = (
                                  signal.dtype == complex),
                              random_state = random_state),
                              ))    
    
    elif method is 'signal_like':
        return np.concatenate((
                    signal_like_noise(signal, 
                                      snr=snr,
                                      length = pad_width[0],
                                      random_state = random_state),
                    signal,
                    signal_like_noise(signal, 
                                      snr=snr,
                                      length = pad_width[1],
                                      random_state = random_state)
                              ))    
    
    else:
        return ValueError('uncorrect value')

