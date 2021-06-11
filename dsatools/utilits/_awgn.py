import numpy as np


__all__ = ['awgn','wgn','wgn_with_snr','awgnDB','signal_like_noise']

#---------------------------------------------------------
def awgnDB(signal, snr_db = 20, random_state = None):
    '''
    Add white gaussian noises corresponding 
        to the target signal-to-noise ratio.

    Parameters
    ----------
    * signal: 1d ndarray.
    * snr_db: float,
        signal-to-noise ratio 
        (in db of amplitudes (not power db)).
    * random_state: float,
        random seed state. 
        
    Returns
    --------
    * signal+noises: 1d ndarray.
    
    Notes
    ---------
    * dB of amplitudes: SNRdB=20log_10(SNR).
    '''    
    return awgn(signal, snr=snr_db, units = 'db')
#--------------------------------------------------------
def wgn(noise_power, length,
        is_complex = False, random_state = None):
    '''
    White Gaussian Noises
    ..math::
        noise = sqrt(noise_power)*radom_normal_samples.
      where radom_normal_samples are samples of 
       normal distribution with mean 0 and variance 1.
    
    Parameters
    ----------
    * noise_power: float,
        noise power (square of it max value).
    * length: int,
        number of noise samples.    
    * is_complex: bool,
        if complex, then complex-valued noise 
        will be returned with independent complex
        and real-valued parts.
    * random_state: float,
        random seed state.    
        
    Returns
    --------
    * noises: 1d ndarray.
    
    '''
    if random_state is not None:
        np.random.seed = random_state
    
    if (is_complex):
        wgn = np.sqrt(noise_power/2)*\
            (np.random.normal(size = length) + \
             1j* np.random.normal(size = length))

    else:    
        wgn = np.sqrt(noise_power)*\
            np.random.normal(size = length)

    return wgn

#------------------------
def wgn_with_snr(signal, snr, length = 1,
                 units = 'db', random_state = None):
    '''
    White Gaussian Noises corresponding 
        to the target signal-to-noise ratio.
    ..math::
        noise = sqrt(noise_power)*radom_normal_samples.
      where 
       * radom_normal_samples are samples of 
         normal distribution with mean 0 and variance 1;
       * noise_power calculated with SNR.
       
    Parameters
    ----------
    * signal: 1d ndarray.
    * snr_db: float,
        signal-to-noise ratio 
        (in db of amplitudes (not power db)).
    * length: number of noise samples.    
    * units: string,
        units = {'linear','bd','bdw','bdm'}. 
    * random_state: float,
        random seed state. 
        
    Returns
    --------
    * noises: 1d ndarray.
    
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
    
    return wgn(noise_power, 
               length = length, 
               is_complex = (signal.dtype == complex), 
               random_state = random_state)

#--------------------------------------    
def awgn(signal, snr, units = 'db', random_state = None):
    '''
    Add white gaussian noises corresponding 
        to the target signal-to-noise ratio.
    ..math::
        out = signal + noises
        noise = sqrt(noise_power)*radom_normal_samples.
      where 
       * radom_normal_samples are samples of 
         normal distribution with mean 0 and variance 1;
       * noise_power calculated with SNR.
    Parameters
    ----------
    * signal: 1d ndarray.
    * snr_db: float,
        signal-to-noise ratio 
        (in db of amplitudes (not power db)).
    * units: string,
        units = {'linear','bd','bdw','bdm'}. 
    * random_state: float,
        random seed state.
        
    Returns
    --------
    * signal+noises: 1d ndarray.
    
    Notes
    ---------
    * dB of amplitudes: SNRdB=20log_10(SNR).
    '''
    signal = np.asarray(signal)
    return signal + wgn_with_snr(signal = signal, 
                        snr = snr, 
                        length = signal.shape[0], 
                        units = units, 
                        random_state = random_state)

#---------------------------------------------------------
def signal_like_noise(signal, 
                      snr=0,
                      length = None,
                      units = 'db',
                      random_state = None):
    '''
   Noises with the signal -like distribution
     corresponding to the target 
     signal-to-noise ratio.
    
    Parameters
    ----------
    * signal: 1d ndarray.
    * snr: float,
        signal-to-noise ratio 
        (in db of amplitudes (not power db)).
    * length, int or None,
        length of noise vector,
        if None, length =  signal.size.   
    * units: string,
        units = {'linear','bd','bdw','bdm'}. 
    * random_state: float,
        random seed state.
    
    Returns
    --------
    * signal+noises: 1d ndarray.
    
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

    if length is None:
        length = signal.shape[0]
    
    signal_power = np.sum(np.square(np.abs(signal)))/signal.size
    noise_power  = signal_power/snr
    

    if (signal.dtype == complex):
        noise = np.sqrt(noise_power/2)*(
                                        np.random.choice(signal.real, 
                                                         size=length, 
                                                         replace=True)
                                        +
                                        np.random.choice(signal.imag, 
                                                         size=length, 
                                                         replace=True)
                                        )
    
    else:
        noise = np.sqrt(noise_power)*np.random.choice(signal, 
                                                      size=length, 
                                                      replace=True)
    
    return noise
#-------------------------------------------------------------------
'''def awgn(signal, snr, units = 'db', random_state = None):

    Add white gaussian noises corresponding 
        to the target signal-to-noise ratio.
    
    Parameters
    ----------
    * signal: 1d ndarray.
    * snr_db: float,
        signal-to-noise ratio 
        (in db of amplitudes (not power db)).
    * units: string,
        units = {'linear','bd','bdw','bdm'}. 
    * random_state: float,
        random seed state.
        
    Returns
    --------
    * signal+noises: 1d ndarray.
    
    Notes
    ---------
    * dB of amplitudes: SNRdB=20log_10(SNR).

  
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
'''

