import numpy as np
import scipy

from ._simsig_tools import _check_list,_rand_uniform
from ._generator_base import generator_base

#------------------------------------------------------------------------------------ 
__all__=['beatsig','Beatsig']

#------------------------------------------------------------------------------------ 
_SIGNAL_PARAMETER_DEFAULT = {'amp':1,'delay':10,
                             'delta_delay':0, 'phase0':0,'callback': None} 

_SYSTEM_PARAMETER_DEFAULT = {'f0':1,'delta_f':0.5,'Tm':51.2, 'fs':10, 'length':512}

#------------------------------------------------------------------------------------ 
def beatsignal(amplitude   = [1],
               delay       = [10.5],
               delta_delay = [0.0],
               phase0      = [0],
               callback    = [None],
               f0      = 1,
               delta_f = 0.5,
               fs      = 10,
               length  = 512,
               Tm      = None,
               snr_db  = None):    
    '''
    Beat harmonic signal generation. The beat signal 
        is result of frequency modulated continium 
        waves (FMCW) processing based on the so-called
        heterodyne tranciver scheme.

    Parameters
    --------------
    * amp: 1d ndarray,
        amplitude of signal components.
    * delay: 1d ndarray,
        delay of fmcw signal components.
    * delta_delay: 1d ndarray,
        fmcw signal delay band (in the case of some 
        changes during the signal durations).                                                        
    * phase0: 1d ndarray,  
        initla phase, does not connected with delay.
    * callback: 1d ndarray, 
        callback for special operations on signals.  
    * f0:  float,     
        initial frequency (carried frequency).
    * delta_f: float, 
        frequency deviation (frequency band).
    * fs: float,      
        is the sampling frequency.
    * length: int,  
        is the signal length. 
    * Tm: float, 
        period of modulation, 
        (N/fs if None).
    * snr_db: float,
        is signa-to-noise ratio (in dB).
   
    Returns
    ----------
    * beat signal: 1d ndarray (complex),
        complex (or analytical) signal.
    
    Notes
    ----------- 
    * f0, Delta_f, fs and N are system parameters 
      should be uiniqly set.
    * Delta_tau represent some delay changes, 
      i.e. target changes during corresponding 
      FMCW signal emittion.          
    * Simulated beat signal has the following
      expression:
      .. math::         
      s= sum{f_i(a_i*exp[j[W(t)*tau_i+W2(t)*Delta_tau_i+phi_0_i])}
                                                            +noises,       
      where:
      * i = 0,.., are the numbers of signals 
      in the superposition (atually number of set dealys).      
      * W(t)   = Delta_f *t/T_m + f_0.
      * W_2(t) = Delta_f *t^2/T_m + f_0.
      * f_0 is the initial frequency;
      * Delta_f is the frequency band 
        (from f_0 to f_0+Delta_f);
      * N   is length (size) of signals samples;
      * t   is the time (up to N/fs);
      * fs  is the sampling frequency;                     
      * a_i         is the amplitude;
      * tau_i       is the signal delay;
      * Delta_tau_i is the time delay band;                   
      * phi_0_i     is the initial phase;
      * f_i         is the modulation callback;
      * noises      are the gaussian white noises.

    Examples
    ----------------
        import dsatools.generator
        from dsatools.generator import callbacks
        import dsatools.utilits as ut

        #Example1
        signal = dsatools.generator.beatsignal()
        ut.probe(signal)

        #Example2
        signal = dsatools.generator.beatsignal(amplitude=[1,2],
                                                delay=[10.5, 23],
                                                delta_delay=[0.0, 0.8],
                                                phase0=[0],
                                                callback=[None],
                                                f0=1,
                                                delta_f=0.5,
                                                fs=10,
                                                length=512,
                                                Tm=None,
                                                snr_db=20)
        ut.probe(signal)

        #Example3
        cb1 = callbacks.harmonic_modulataion(amp_am=0.6,freq_am=0.5,phase_shift=0)
        cb2 = callbacks.harmonic_modulataion(amp_am=1,freq_am=0.3,phase_shift=0)

        signal = dsatools.generator.beatsignal(amplitude=[1,0.5,1, 0.6, 0.7],
                                                delay=[10.5, 23,83, 83, 96],
                                                delta_delay=[0.0, 0.8,0.1,0.9],
                                                phase0=[0,0, 1.3],
                                                callback=[cb1,cb1, None, cb2],
                                                f0=1,
                                                delta_f=0.5,
                                                fs=10,
                                                length=512,
                                                Tm=None,                                   
                                                snr_db=20)
        ut.probe(signal)
    '''    
        
    if (Tm is None):
        Tm = length/fs
    
    signal = Beatsignal(f0,delta_f,Tm,fs,length)
    
    signal.set_signal_parameters(amplitude   = amplitude,
                                 delay       = delay,
                                 delta_delay = delta_delay,
                                 phase0      = phase0,
                                 callback    = callback)
    
    return signal.get_signal(snr_db = snr_db) 

#------------------------------------------------------------------------------------ 
class Beatsignal(generator_base):
    '''
    Beat harmonic signal generation.

    Atriburts
    -------------    
    > system_parameters:
        {f0, delta_f, Tm, fs, length}, 
      where:
    * f0: float,
        initial frequency (carried frequency).
    * delta_f: float,
        frequency deviation (frequency band).
    * Tm: float,
        period of modulation.
    * fs: float,
        is the sampling frequency.
    * length: int,
       is the signal length.            

    > signal_parameters: 
        {amp, delay, delta_delay, phase0, callback},
      where:
      * amp: 1d ndarray, 
          amplitude of signals.
      * delay:   
          fmcw signal delay;
      * delta_delay: 1d ndarray,   
          fmcw signal delay band (in the case of 
          some changes during the signal durations);                                                        
      * phase0: 1d ndarray,
          initla phase, does not connected with delay;
      * callback: 1d ndarray, 
          callback for special operations on signals.  

    Methods
    ---------
    * set_system_parameters.
    * get_system_parameters.
    * set_signal_parameters.
    * add_signal_parameters.
    * print_signal_parameters.
    * get_signal.
   
    Notes
    ----------- 
    * f0, Delta_f, fs and N are system parameters 
      should be uiniqly set.
    * Delta_tau represent some delay changes, 
      i.e. target changes during corresponding 
      FMCW signal emittion.          
    * Simulated beat signal has the following
      expression:
      .. math::         
      s= sum{f_i(a_i*exp[j[W(t)*tau_i+W2(t)*Delta_tau_i+phi_0_i])}
                                                            +noises,       
      where:
      * i = 0,.., are the numbers of signals 
      in the superposition (atually number of set dealys).      
      * W(t)   = Delta_f *t/T_m + f_0.
      * W_2(t) = Delta_f *t^2/T_m + f_0.
      * f_0 is the initial frequency;
      * Delta_f is the frequency band 
        (from f_0 to f_0+Delta_f);
      * N   is length (size) of signals samples;
      * t   is the time (up to N/fs);
      * fs  is the sampling frequency;                     
      * a_i         is the amplitude;
      * tau_i       is the signal delay;
      * Delta_tau_i is the time delay band;                   
      * phi_0_i     is the initial phase;
      * f_i         is the modulation callback;
      * noises      are the gaussian white noises.

    Examples
    ----------------
        import dsatools.generator 
        from dsatools.generator import callbacks
        import dsatools.utilits as ut
        cb1 = callbacks.harmonic_modulataion(amp_am=0.1,freq_am=0.5,phase_shift=0)
        callbacks.probe_modulation(cb1,512)
        sig1 = dsatools.generator.Beatsignal()
        sig1.get_system_parameters()
        sig1.set_signal_parameters(amplitude   = [1,0.5],
                                   delay       = [10.5],
                                   delta_delay = [0.0,0.0],
                                   phase0      = [0],
                                   callback    = [cb1])
        signal = sig1.get_signal(snr_db = 20)
        ut.probe(signal)  
    '''
    #@override 
    def __init__(self,
                 f0      = _SYSTEM_PARAMETER_DEFAULT['f0'],
                 delta_f = _SYSTEM_PARAMETER_DEFAULT['delta_f'],
                 Tm      = _SYSTEM_PARAMETER_DEFAULT['Tm'],
                 fs      = _SYSTEM_PARAMETER_DEFAULT['fs'],
                 length  = _SYSTEM_PARAMETER_DEFAULT['length']                 
                ):
        
        self._signal_parameters_dict_default = _SIGNAL_PARAMETER_DEFAULT.copy()
        self._system_parameters_dict_default = _SYSTEM_PARAMETER_DEFAULT.copy()
        
        #here!
        self.set_system_parameters(f0, delta_f, Tm, fs, length)
        self.set_signal_parameters_dict_default()
    
    #------------------------------------------------------------------------------------ 
    #@override 
    def set_system_parameters(self,
                              f0      = _SYSTEM_PARAMETER_DEFAULT['f0'],
                              delta_f = _SYSTEM_PARAMETER_DEFAULT['delta_f'],
                              Tm      = _SYSTEM_PARAMETER_DEFAULT['Tm'],
                              fs      = _SYSTEM_PARAMETER_DEFAULT['fs'],
                              length  = _SYSTEM_PARAMETER_DEFAULT['length'] ):
        '''
        Set system parameters.
              
        Parameters
        -----------
        * f0: float,
            initial frequency (carried frequency).
        * delta_f: float,
            frequency deviation (frequency band).
        * Tm: float,
            period of modulation.
        * fs: float,
            is the sampling frequency.
        * length: int,
           is the signal length.
        
        '''
        self._system_parameters['f0'] = f0
        self._system_parameters['delta_f'] = delta_f
        self._system_parameters['Tm'] = Tm
        self._system_parameters['fs'] = fs
        self._system_parameters['length'] = length 
    
    #------------------------------------------------------------------------------------  
    #@override   
    def make_signal_parameters_dict(self, 
                                    amplitude = _SIGNAL_PARAMETER_DEFAULT['amp'],
                                    delay     = _SIGNAL_PARAMETER_DEFAULT['delay'],
                                    delta_delay = _SIGNAL_PARAMETER_DEFAULT['delta_delay'],
                                    phase0    = _SIGNAL_PARAMETER_DEFAULT['phase0'],
                                    callback  = _SIGNAL_PARAMETER_DEFAULT['callback']):
        '''
        Make signal_parameters dictionary.
            
        Parameters
        ----------
        * amp: 1d ndarray, 
            amplitude of signals.
        * delay:   
            fmcw signal delay;
        * delta_delay: 1d ndarray,   
            fmcw signal delay band (in the case of 
            some changes during the signal durations);                                                        
        * phase0: 1d ndarray,
            initla phase, does not connected with delay;
        * callback: 1d ndarray, 
            callback for special operations on signals.

        Returns
        --------
        * signal_parameters_dict: dict,
            signal parameters dictionary.    
        
        ''' 
        signal_parameters_dict = self.get_signal_parameters_dict_default()
        
        #here!
        signal_parameters_dict['amp']      = amplitude
        signal_parameters_dict['delay']    = delay
        signal_parameters_dict['delta_delay'] = delta_delay
        signal_parameters_dict['phase0']   = phase0
        signal_parameters_dict['callback'] = callback
        
        return signal_parameters_dict    

    #------------------------------------------------------------------------------------         
    #@override  
    def add_signal_parameters(self, 
                              amplitude   = [_SIGNAL_PARAMETER_DEFAULT['amp']],
                              delay       = [_SIGNAL_PARAMETER_DEFAULT['delay']],
                              delta_delay = [_SIGNAL_PARAMETER_DEFAULT['delta_delay']],
                              phase0      = [_SIGNAL_PARAMETER_DEFAULT['phase0']],
                              callback    = [_SIGNAL_PARAMETER_DEFAULT['callback']]):
        '''
        Add signal_parameters.

        Parameters
        --------------
        * amplitude: 1d ndarray, 
            amplitudes of signal component.
        * delay: 1d ndarray, 
            signal component delays.
        * delta_delay: 1d ndarray,
            fmcw signal delay band 
            (in the case of some changes 
            during the signal durations).            
        * phase0: 1d ndarray,
            initla phase of signal component.
        * callback: 1d ndarray,
            callbacks for special operations on signals.
                    
        Notes
        -------------
        * Formats of the input: float, list, tuple.
        * in the case of different length of array, 
            all will be resized to delays length.
        
        '''         
        # main array - f0
        delay     = _check_list(delay,-1)         
        len_list = len(delay) #required length for all other arrays       
        
        amplitude   = _check_list(amplitude,   len_list, 'last')
        delta_delay = _check_list(delta_delay, len_list, 0)
        phase0      = _check_list(phase0,      len_list, 0)        
        callback    = _check_list(callback,    len_list, 'None')  
        
        dict2add = []
        
        #here!
        #TODO: some optimizations will be valuable        
        for (amplitude_,
             delay_,
             delta_delay_,
             phase0_,
             callback_) in \
                        zip(amplitude,
                            delay,
                            delta_delay,
                            phase0,
                            callback):            
            
            #here!
            dict2add += [self.make_signal_parameters_dict(amplitude_, 
                                                          delay_,
                                                          delta_delay_,
                                                          phase0_,
                                                          callback_)]        
            
        
        self.add_signal_parameters_dicts(dict2add)    
    
    #------------------------------------------------------------------------------------         
    #@override  
    def set_signal_parameters(self, 
                              amplitude   = [_SIGNAL_PARAMETER_DEFAULT['amp']],
                              delay       = [_SIGNAL_PARAMETER_DEFAULT['delay']],
                              delta_delay = [_SIGNAL_PARAMETER_DEFAULT['delta_delay']],
                              phase0       = [_SIGNAL_PARAMETER_DEFAULT['phase0']],
                              callback    = [_SIGNAL_PARAMETER_DEFAULT['callback']]): 
        '''
        Set signal_parameters.
        
        Parameters
        --------------
        * amplitude: 1d ndarray, 
            amplitudes of signal component.
        * delay: 1d ndarray, 
            signal component delays.
        * delta_delay: 1d ndarray,
            fmcw signal delay band 
            (in the case of some changes 
            during the signal durations).            
        * phase0: 1d ndarray,
            initla phase of signal component.
        * callback: 1d ndarray,
            callbacks for special operations on signals.
                    
        Notes
        -------------
        * Formats of the input: float, list, tuple.
        * in the case of different length of array, 
            all will be resized to delays length.
            
        '''        
        self.clear_signal_parameters()
        
        #here!
        self.add_signal_parameters(amplitude,
                                   delay,
                                   delta_delay,
                                   phase0,
                                   callback)
    
    #------------------------------------------------------------------------------------   
    #@override    
    def _sim_one_sig(self, sig_param): 
        '''
        Simulate one harmonic beat signal.
        
        Parameters
        -------------
        * sig_param: dict 
            dictionary of signal parameters, whcih include
            (a,delay,delta_delat,phi0,callback).
            
        Returns 
        ------------
        * sig: 1d ndarray (complex),
            simulated signal.
        
        Notes
        ---------
        * Parameters f0, Delta_f, fs and N are system parameters.
        * Simulate one harmonic beat signal in form:
          s = f(a*exp[j[W(t)*tau + W_2(t)*Delta_tau +phi_0]),    
          where:
          * W(t)   = 2pi(Delta f *t/T_m + f_0);
          * W_2(t) = 2pi(Delta f *t^2/T_m + f_0);
          * a is the amplitude;
          * f_0 is the initial frequency;
          * tau is the signal delay;
          * Delta_f is the frequency band (from f_0 to f_0+Delta_f);
          * Delta_au is the time delay band
          * N is length (size) of signals samples;
          * t  is the time (up to N/fs);
          * fs is the sampling frequency;                    
          * phi_0 is the initial phase;
          * f modulation callback.
        
        '''
        f0   = self._system_parameters['f0']
        incF = self._system_parameters['delta_f'] 
        Tm   = self._system_parameters['Tm'] 
        fs   = self._system_parameters['fs'] 
        N    = self._system_parameters['length']  

        delay       = sig_param['delay'] 
        delta_delay = sig_param['delta_delay'] 
        A           = sig_param['amp']        
        phi0        = sig_param['phase0']
        callback    = sig_param['callback']

        t   = np.arange(N)/fs 
        W   = 2*np.pi*(incF*t/Tm+f0)
        W2  = 2*np.pi*(incF*t**2/Tm+f0)
        
        sig = A*np.exp(1j*(W*delay + W2* delta_delay+ phi0)) 
        
        sig = np.asarray(sig,dtype= np.complex)
        
        if (callback in  ['None', None]):
              return sig

        elif type(callback ) is not list:
            callback = list([callback])

            for callback_i in callback:
                sig = callback_i(sig)
           
            return sig   

    #------------------------------------------------------------------------------------         
    #@override  
    def add_random_signal_parameters(self, 
                                     n_of_params = 1,
                                     amplitude_range   = [0,_SIGNAL_PARAMETER_DEFAULT['amp']],
                                     delay_range       = [0,_SIGNAL_PARAMETER_DEFAULT['delay']],
                                     delta_delay_range = [0,_SIGNAL_PARAMETER_DEFAULT['delta_delay']],
                                     phase0_range      = [0,_SIGNAL_PARAMETER_DEFAULT['phase0']],):  
        '''
        Add random uniformly distributed signal_parameters.
   
        Parameters
        ----------
        * n_of_params: int,
            number of paramentrs.
        * amplitude_range: [float,float],  
            ranges of amplitudes.
        * delay_range: [float,float],  
            ranges of the signal delays.
        * delta_delay_range: [float,float],  
            ranges of the signal delta delays.
        * phase0_range: [float,float], 
            ranges of the initla phases.
          
        Notes
        ---------
        * Callbacks doesnot applied for this function.
        
        '''
        scale_float = _SCALE_TO_FLOAT_
        
        amplitude   = _rand_uniform(amplitude_range,   n_of_params, scale_float)
        delay       = _rand_uniform(delay_range,       n_of_params, scale_float)
        delta_delay = _rand_uniform(delta_delay_range, n_of_params, scale_float)
        phase0      = _rand_uniform(phase0_range,      n_of_params, scale_float)        
        
        #here!
        self.add_signal_parameters(amplitude,
                                   delay,
                                   delta_delay,
                                   phase0,
                                   callback = n_of_params * [None])    
   
    
    #------------------------------------------------------------------------------------   
    def get_set_signal(self, amplitude, delay, delta_delay, phase0, callback, snr_db):
        '''
        For test:
        Method for set and get signal
        
        '''
        
        self.set_signal_parameters(amplitude   = amplitude,
                                   delay       = delay,
                                   delta_delay = delta_delay,
                                   phase0      = phase0,
                                   callback    = callback)
    
        return self.get_signal(snr_db = snr_db)     
    
    #------------------------------------------------------------------------------------   
