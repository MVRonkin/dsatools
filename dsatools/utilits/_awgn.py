import numpy as np
import scipy

__all__ = ['awgn','awgnDB','signal_like_noise']

#-------------------------------------------------------------------
def awgnDB(signal, snr_db = 20, random_state = None):
    '''
    Add white gaussian noises corresponding 
        to the target signal-to-noise ratio.

    Parameters
    ----------
    * x: 1d ndarray.
    * snr_db: float,
        signal-to-noise ratio 
        (in db of amplitudes (not power db)).
    * random_state: float,
        random seed state. 
        
    Returns
    --------
    * x+noises: 1d ndarray.
    
    Notes
    ---------
    * dB of amplitudes: SNRdB=20log_10(SNR).
    '''    
    return awgn(signal, snr=snr_db, units = 'db')

#-------------------------------------------------------------------
def awgn(signal, snr, units = 'db', random_state = None):
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
    * random_state: float,
        random seed state.
        
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
    
    if random_state is not None:
        np.random.seed = random_state
    
    signal = np.asarray(signal)
    
    signal_power = np.sum(np.square(np.abs(signal)))/signal.size
    noise_power  = signal_power/snr
    
    #TODO: is ther complex normal distribution exist?
    if (signal.dtype == complex):
        wgn = np.sqrt(noise_power/2)*\
            (np.random.normal(size = signal.shape) + \
             1j* np.random.normal(size = signal.shape))
    
    else:    
        wgn = (np.sqrt(noise_power))*(
            np.random.normal(size = signal.shape))
    
    return signal + wgn
#-------------------------------------------------------------------
def signal_like_noise(signal, snr=0, units = 'db', random_state = None):
    '''
   Noises with the signal -like distribution
     corresponding to the target 
     signal-to-noise ratio.
    
    Parameters
    ----------
    * x: 1d ndarray.
    * snr_db: float,
        signal-to-noise ratio 
        (in db of amplitudes (not power db)).
    * units: string,
        units = {'linear','bd','bdw','bdm'}. 
    * random_state: float,
        random seed state.
    
    Returns
    --------
    * x+noises: 1d ndarray.
    
    Notes
    ---------
    * dB of amplitudes: SNRdB=20log_10(SNR).

    ''' 
    if random_state is not None:
        np.random.seed = random_state
    
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
    
    signal_power = np.sum(np.square(np.abs(signal))
                         )/signal.size
    noise_power  = signal_power/snr
    

    if (signal.dtype == complex):
        
        idxs_real    = np.random.randint(0,
                                         signal1.shape[0],
                                         signal1.shape[0])
        
        idxs_complex = np.random.randint(0,
                                         signal1.shape[0],
                                         signal1.shape[0])
        
        noise = np.sqrt(noise_power/2)*(signal.real[idxs_real] +  
                                        1j*signal.imag[idxs_complex])
    
    else:
        idxs = np.random.randint(0,signal1.shape[0],signal1.shape[0])                   
        noise = (np.sqrt(noise_power))*signal[idxs]
    
    return noise