import numpy as np
from . _awgn import wgn_with_snr, signal_like_noise

_all__ = ['pad_to_power_of_2']

def pad_to_power_of_2(signal,
                      method = None, param = 0):
    '''
    Padding to length as power of 2.
    
    Paramteres
    -----------------
    * signal: 1d ndarray,
        input signal.
    * method, string,
        method of padding:
        * 'zeroing' - with 0;
        * None or 'constant'
            - with param value;
        * 'reflect', with the the mirrored 
                        samples, without last.
        * 'symmetric', with the the mirrored 
                        samples, with last.
        * 'cyclic', with repeat from first sample.
        * 'linear_ramp' with samples from 
                last value to 0
        * 'noise','wgn': with
            noise of power param;
        * 'awgn': noise of SNR param; 
        * 'signal_noise': with noise as
           random signal samples 
           with SNR param;
  
        
    '''       
    N = len(signal)
    
    N_new = np.power(2,int(np.log2(N))+1)
    
    if method is None or method is 'constant':
        return np.pad(signal,(0,N_new-N),
                      'constant', 
                      constant_values = param)

    elif method is 'zeroing':
        return np.pad(signal,(0,N_new-N),'constant', 
                                          constant_values = 0)
    
    elif method in ['noise','wgn']:
        return np.concatenate((signal, 
                               param*np.random.randn(N_new-N) ))
    
    elif method is 'awgn':
        return np.concatenate((signal, 
                               wgn_with_snr(signal, 
                                               param, 
                                               length=N_new-N) )) 
    
    elif method is 'signal_noise':
        return np.concatenate((signal, 
                               signal_like_noise(signal,
                                                 param)[:N_new-N]))    
    
    elif method in ['reflect','symmetric','linear_ramp']:
        return np.pad(signal,(0,N_new-N),method)
    
    elif method is 'cyclic':
        #TODO: what if N_new-N>N - it is not the case for pow2
        return np.concatenate((signal, 
                               signal[:N_new-N] ))    
    
    else:
        return ValueError('uncorrect value')

