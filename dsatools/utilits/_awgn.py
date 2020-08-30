import numpy as np
import scipy

__all__ = ['awgn','awgnDB']

#-------------------------------------------------------------------
def awgnDB(signal, snr_db = 20):
    '''
    Add white gaussian noises corresponding 
        to the target signal-to-noise ratio.

    Parameters
    ----------
    * x: 1d ndarray.
    * snr_db: float,
        signal-to-noise ratio 
        (in db of amplitudes (not power db)).
      
    Returns
    --------
    * x+noises: 1d ndarray.
    
    Notes
    ---------
    * dB of amplitudes: SNRdB=20log_10(SNR).
    '''    
    return awgn(signal, snr=snr_db, units = 'db')

#-------------------------------------------------------------------
def awgn(signal, snr, units = 'db'):
    '''
    Add white gaussian noises corresponding 
        to the target signal-to-noise ratio.
    
    Parameters
    ----------
    * x: 1d ndarray.
    * snr_db: float,
        signal-to-noise ratio 
        (in db of amplitudes (not power db)).
    * units: string,
        units = {'linear','bd','bdw','bdm'}. 
    
    Returns
    --------
    * x+noises: 1d ndarray.
    
    Notes
    ---------
    * dB of amplitudes: SNRdB=20log_10(SNR).

    '''    
    if (units == 'db'):
        snr = np.power(10,snr/10) # half of snr due to power
    elif (units == 'dbw'):
        snr = np.power(10,snr/5) # half of snr due to power
    elif (units == 'dbm'):
        snr = np.power(10,(snr-30)/5) # half of snr due to power    
    elif (units == 'linear'):
        pass
    else:
        raise ValueError('undefined untis')
        
    signal = np.asarray(signal)
    
    signal_power = np.sum(np.square(np.abs(signal)))/signal.size
    noise_power  = signal_power/snr
    
    #TODO: is ther complex normal distribution exist?
    if (signal.dtype == np.complex):
        wgn = (np.sqrt(noise_power/2))*\
            (np.random.normal(size = signal.shape) + \
             1j* np.random.normal(size = signal.shape))
    
    else:    
        wgn = (np.sqrt(noise_power))*(np.random.normal(size = signal.shape))
    
    return signal + wgn

