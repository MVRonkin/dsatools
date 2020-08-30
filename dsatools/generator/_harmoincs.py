import numpy as np
import scipy

from ._simsig_tools import _check_list,_rand_uniform
from ._generator_base import generator_base

#------------------------------------------------------------------------------------ 
__all__=['harmonics','Harmonics']

#------------------------------------------------------------------------------------ 
_SIGNAL_PARAMETER_DEFAULT = {'amp':1, 'f0':1, 'delta_f':0, 'delay':0,'phase0':0,'callback': None} 
_SYSTEM_PARAMETER_DEFAULT = {'fs':10, 'length':512}
#------------------------------------------------------------------------------------ 
def harmonics(amplitude = [1],
              f0        = [1],
              delta_f   = [0],
              delay     = [0],
              phase0    = [0],
              callback  = [None],
              fs=10, 
              length=512, 
              snr_db = None):
    '''
    Harmonic signal generation.
      
    Parameters
    ------------
    * amplitude: 1d ndarray,
        amplitude of signals.
    * f0:  1d ndarray,
        initial frequency (carried frequency).
    * delta_f: 1d ndarray, 
        delta_f frequency (frequency band).
    * delay: 1d ndarray,   
        signal delay.
    * phase0: 1d ndarray,  
        initla phase.
    * callback: 1d ndarray,
        callback for special operations on signals.  
    * fs: float,    
        is the sampling frequency.
    * length: int,
        is the signal length;
    * snr_db: float,
        sngnal-to-noise ration in dB.
        
    Returns:
    -------------
    * signal: 1d ndarray (complex), 
        harmonic signal.
    
     
    Notes
    ---------
    * Fs and N are the system parameters.        
    * Simulate harmonic (actually frequency modulated signal) 
      in the following form:
      ..math::        
      s = sum{f_i(a_i*exp[j2pi(f_0_i(t-tau_i)+
          Delta_f_i(t-tau_i)^2/(N/fs))+j varphi_0_i])}+noises, 
      where:
      * i = 0,.., are the signals number in superposition
        (actually the number of the set initial frequencies(f0));
      * a_i    is the amplitude;
      * f_0_i  is the initial frequency;
      * tau_i is the signal delay;
      * Delta f_i  is the frequency band (from f_0 to f_0+Delta_f);
      * varphi_0_i is the initial phase
      * f_i is the modulation callback;
      * t   is the time (up to N/fs);
      * N   is length (size) of signals samples;                    
      * fs  is the sampling frequency;                    
      * noises are the gaussian white noises.

    Example
    -----------
        import dsatools.generator 
        from dsatools.generator import callbacks
        import dsatools.utilits as ut
        
        #Example1----------------------------------------
        signal = dsatools.generator.harmonics()
        ut.probe(signal)

        #Example2----------------------------------------
        signal = dsatools.generator.harmonics(amplitude=[1],
                                               f0=[1,2,3],
                                               delta_f=[0.3],
                                               delay=[0],
                                               phase0=[0],
                                               callback=[None],
                                               fs=10,
                                               length=512,
                                               snr_db=None,)
        ut.probe(signal)

        #Example3----------------------------------------
        cb1 = callbacks.harmonic_modulataion(amp_am=0.5,freq_am=9.5,phase_shift=0)
        cb2 = callbacks.harmonic_modulataion(amp_am=0.7,freq_am=8.2,phase_shift=0)

        signal = dsatools.generator.harmonics(amplitude=[1,1,0.4,0.3],
                                               f0=[1,2,3,4],
                                               delta_f=[0.2,1.3,],
                                               delay =[0,0,0,4],
                                               phase0=[0,1.2],
                                               callback=[cb1,None,cb2],
                                               fs=10,
                                               length=512,
                                               snr_db=20,)
        ut.probe(signal)
    '''    
    
    signal = Harmonics(fs, length)
    signal.set_signal_parameters(amplitude = amplitude,
                                 f0       = f0,
                                 delta_f  = delta_f,
                                 delay    = delay,
                                 phase0   = phase0,
                                 callback = callback,)
    
    return signal.get_signal(snr_db = snr_db)

#------------------------------------------------------------------------------------ 
class Harmonics(generator_base):
    '''
    Harmonic signal generation.
    
    Atriburts
    ----------------
   > system_parameters = {fs, length}, 
   * fs: float,    
       is the sampling frequency.
   * length: int, 
       is the signal length.
   > signal_parameters = list of
       {amp,f0,delta_f,delay,phase0,callback},
   * amplitude: 1d ndarray,
       amplitude of signal components.
   * f0: 1d ndarray,
       initial components frequency 
       (carried frequency).
   * delta_f: 1d ndarray,   
       delta_f frequency components band.
   * delay: 1d ndarray,   
       signal components delay.
   * phase0: 1d ndarray,  
       initla phase of components.
   * callback: 1d ndarray,
       callback for special operations on signals.  
        
    Methods
    -----------
    * set_system_parameters;
    * get_system_parameters;
    * set_signal_parameters;
    * add_signal_parameters;
    * print_signal_parameters;
    * get_signal.       
      
    Notes
    ---------
    * Fs and N are the system parameters.        
    * Simulate harmonic (actually frequency modulated signal) 
      in the following form:
      ..math::        
      s = sum{f_i(a_i*exp[j2pi(f_0_i(t-tau_i)+
          Delta_f_i(t-tau_i)^2/(N/fs))+j varphi_0_i])}+noises, 
      where:
      * i = 0,.., are the signals number in superposition
        (actually the number of the set initial frequencies(f0));
      * a_i    is the amplitude;
      * f_0_i  is the initial frequency;
      * tau_i is the signal delay;
      * Delta f_i  is the frequency band (from f_0 to f_0+Delta_f);
      * varphi_0_i is the initial phase
      * f_i is the modulation callback;
      * t   is the time (up to N/fs);
      * N   is length (size) of signals samples;                    
      * fs  is the sampling frequency;                    
      * noises are the gaussian white noises.

    Example
    -----------
        import dsatools.generator 
        from dsatools.generator import callbacks
        import dsatools.utilits as ut

        cb1 = callbacks.harmonic_modulataion(amp_am=0.1,freq_am=0.5,phase_shift=0)
        callbacks.probe_modulation(cb1,512)
        cb2 = callbacks.pulse_modulataion(200,400)
        callbacks.probe_modulation(cb2,512)

        signal1 = dsatools.generator.Harmonics()
        signal1.get_system_parameters()
        signal1.set_signal_parameters(amplitude=[1,0.5],
                                      f0=[1,2,3],
                                      delta_f=[0.4,0.1],
                                      delay=[0],
                                      phase0=[0],
                                      callback=[cb1,cb2],)
        sig1 = signal1.get_signal(snr_db = 200)
        ut.probe(sig1)
    '''
    #@override 
    def __init__(self, 
                 fs     = _SYSTEM_PARAMETER_DEFAULT['fs'],
                 length = _SYSTEM_PARAMETER_DEFAULT['length'] 
                ):
        
        self._signal_parameters_dict_default = _SIGNAL_PARAMETER_DEFAULT.copy()
        self._system_parameters_dict_default = _SYSTEM_PARAMETER_DEFAULT.copy()
        
        self.set_system_parameters(fs, length)
        self.set_signal_parameters_dict_default()
    
    #------------------------------------------------------------------------------------ 
    #@override 
    def set_system_parameters(self, 
                             fs=_SYSTEM_PARAMETER_DEFAULT['fs'], 
                             length = _SYSTEM_PARAMETER_DEFAULT['fs']):
        '''
        Set system parameters.
        
        Parameters
        -------------
        * fs: float,
            is the sampling frequency.
        * length: int,
            is the length of signal.
        '''
        self._system_parameters['fs'] = fs
        self._system_parameters['length'] = length 
    
    #------------------------------------------------------------------------------------  
    #@override   
    def make_signal_parameters_dict(self, 
                                    amplitude = _SIGNAL_PARAMETER_DEFAULT['amp'],
                                    f0        = _SIGNAL_PARAMETER_DEFAULT['f0'],
                                    delta_f   = _SIGNAL_PARAMETER_DEFAULT['delta_f'],
                                    delay     = _SIGNAL_PARAMETER_DEFAULT['delay'],
                                    phase0    = _SIGNAL_PARAMETER_DEFAULT['phase0'],
                                    callback  = _SIGNAL_PARAMETER_DEFAULT['callback']):
        '''
        Make the signal parameters dictionary.
        
        Parameters
        ------------
        * amplitude: 1d ndarray,
            amplitude of signal components.
        * f0: 1d ndarray,
            initial components frequency 
            (carried frequency).
        * delta_f: 1d ndarray,   
            delta_f frequency components band.
        * delay: 1d ndarray,   
            signal components delay.
        * phase0: 1d ndarray,  
            initla phase of components.
        * callback: 1d ndarray,
            callback for special operations on signals. 
        
        Returns
        ----------
        * signal_parameters_dict: dict,
            signal parameters dictionary.
            
        ''' 
        signal_parameters_dict = self.get_signal_parameters_dict_default()
        
        signal_parameters_dict['amp']      = amplitude
        signal_parameters_dict['f0']       = f0
        signal_parameters_dict['delta_f']  = delta_f
        signal_parameters_dict['delay']    = delay
        signal_parameters_dict['phase0']   = phase0
        signal_parameters_dict['callback'] = callback
        
        return signal_parameters_dict    

    #------------------------------------------------------------------------------------         
    #@override  
    def add_signal_parameters(self, 
                              amplitude = [_SIGNAL_PARAMETER_DEFAULT['amp']],
                              f0       = [_SIGNAL_PARAMETER_DEFAULT['f0']],
                              delta_f  = [_SIGNAL_PARAMETER_DEFAULT['delta_f']],
                              delay    = [_SIGNAL_PARAMETER_DEFAULT['delay']],
                              phase0   = [_SIGNAL_PARAMETER_DEFAULT['phase0']],
                              callback = [_SIGNAL_PARAMETER_DEFAULT['callback']]):
        '''
        Add signal parameters.
        
        Parameters
        ------------
        * amplitude: 1d ndarray,
            amplitude of signal components.
        * f0: 1d ndarray,
            initial components frequency 
            (carried frequency).
        * delta_f: 1d ndarray,   
            delta_f frequency components band.
        * delay: 1d ndarray,   
            signal components delay.
        * phase0: 1d ndarray,  
            initla phase of components.
        * callback: 1d ndarray,
            callback for special operations on signals.            
        
        Notes
        ----------
        * formats of the input: float, list, tuple.
        * in the case of different length of array, 
           all will be resized to f0_s length.                      
  
        '''         
        # main array - f0
        f0     = _check_list(f0,-1)         
        len_list = len(f0) #required length for all other arrays       
        
        amplitude = _check_list(amplitude, len_list, 'last')
        delta_f   = _check_list(delta_f,   len_list, 0)
        delay     = _check_list(delay,     len_list, 0)        
        phase0    = _check_list(phase0,    len_list, 0)        
        callback  = _check_list(callback,  len_list, 'None')  
        
        dict2add = []
        for (amplitude_,
             f0_,
             delta_f_,
             delay_,
             phase0_,
             callback_) in \
                zip(amplitude,
                    f0,
                    delta_f,
                    delay,
                    phase0,
                    callback):            
            
            dict2add += [self.make_signal_parameters_dict(amplitude_, 
                                                          f0_,
                                                          delta_f_,
                                                          delay_, 
                                                          phase0_,
                                                          callback_)]        
            
        self.add_signal_parameters_dicts(dict2add)    
    
    #------------------------------------------------------------------------------------         
    #@override  
    def set_signal_parameters(self, 
                              amplitude = [_SIGNAL_PARAMETER_DEFAULT['amp']],
                              f0        = [_SIGNAL_PARAMETER_DEFAULT['f0']],
                              delta_f   = [_SIGNAL_PARAMETER_DEFAULT['delta_f']],
                              delay     = [_SIGNAL_PARAMETER_DEFAULT['delay']],
                              phase0    = [_SIGNAL_PARAMETER_DEFAULT['phase0']],
                              callback  = [_SIGNAL_PARAMETER_DEFAULT['callback']]): 
        '''
        Set signal parameters.
            
        Parameters
        ------------
        * amplitude: 1d ndarray,
            amplitude of signal components.
        * f0: 1d ndarray,
            initial components frequency 
            (carried frequency).
        * delta_f: 1d ndarray,   
            delta_f frequency components band.
        * delay: 1d ndarray,   
            signal components delay.
        * phase0: 1d ndarray,  
            initla phase of components.
        * callback: 1d ndarray,
            callback for special operations on signals.            
        
        Notes
        ----------
        * formats of the input: float, list, tuple.
        * in the case of different length of array, 
           all will be resized to f0_s length.   
           
        '''        
        self.clear_signal_parameters()
        self.add_signal_parameters(amplitude,
                                   f0,
                                   delta_f,
                                   delay,
                                   phase0,
                                   callback)
    
    #------------------------------------------------------------------------------------   
    #@override  
    def add_random_signal_parameters(self, 
                                     n_of_params = 1,
                                     amplitude_range = [0,_SIGNAL_PARAMETER_DEFAULT['amp']],
                                     f0_range        = [0,_SIGNAL_PARAMETER_DEFAULT['f0']],
                                     delta_f_range   = [0,_SIGNAL_PARAMETER_DEFAULT['delta_f']],
                                     delay_range     = [0,_SIGNAL_PARAMETER_DEFAULT['delay']],
                                     phase0_range    = [0,_SIGNAL_PARAMETER_DEFAULT['phase0']]):  
        '''
        Add random uniformly distributed signal_parameters.
 
        Parameters
        -------------
        * n_of_params:  int,   
            number of paramentrs.
        * amplitude_range: [float,float],
            ranges of amplitudes.
        * f0_range: [float,float],       
            ranges of the initial frequencies 
            (carried frequencies).
        * delta_f_range: [float,float],  
            ranges of the delta_f frequencies 
            (frequency bands).
        * delay_range: [float,float],
            ranges of the signal delays.
        * phase0_range: [float,float],   
            ranges of the initla phases.
        
        Notes
        -------
        * Callbacks doesnot applied for this function.            
        
        '''
        scale_float = _SCALE_TO_FLOAT_
        amplitude = _rand_uniform(amplitude_range, n_of_params, scale_float)
        f0        = _rand_uniform(f0_range,        n_of_params, scale_float)
        delta_f   = _rand_uniform(delta_f_range,   n_of_params, scale_float)        
        delay     = _rand_uniform(delay_range,     n_of_params, scale_float)       
        phase0    = _rand_uniform(phase0_range,    n_of_params, scale_float)        
        
        self.add_signal_parameters(amplitude,
                                   f0,
                                   delta_f,
                                   delay,
                                   phase0,
                                   callback = n_of_params * [None])    
   
    
    #------------------------------------------------------------------------------------   
    #@override    
    def _sim_one_sig(self, sig_param): 
        '''
        Simulate one harmonic (actually frequency modulated signal). 

        Parameters
        -----------
        * sig_param: dict,
            dictionary of signal parameters, whcih include
            (a,f_0,\Delta f,\tau,phi0,callback).
        
        Returns
        -----------
        * sig: 1d ndarray (complex),
            simulated signal.
      
        Notes
        ---------
        * Fs and N are system parameters.
        * In harmonic signal \tau and \varphi_0/2/pi 
          are play the same role.
        * If callback is not None: s = callback(s) 
          (format of callback = f(x)),
          if callback is None it does not applied.
        * Signal in form:
          ..math::
          s = f(a*exp[j2pi(f_0(t-tau)+Delta_f(t-tau)^2/(N/fs))+j varphi_0]), 
          where:
          * a   is the amplitude;
          * f_0 is the initial frequency;
          * tau is the signal delay;
          * Delta_f is the frequency band 
            (from f_0 to f_0+\Delta f);
          * N is length (size) of signals samples;
          * fs is the sampling frequency;
          * t  is the time (up to N/fs);
          * varphi_0 is the initial phase
          * f modulation callback.
          
        '''
        fs   = self._system_parameters['fs']
        N    = self._system_parameters['length']

        f0       = sig_param['f0'] 
        incF     = sig_param['delta_f'] 
        tau      = sig_param['delay']        
        phi0     = sig_param['phase0']
        A        = sig_param['amp']
        callback = sig_param['callback']

        t   = np.arange(N)/fs - tau

        Tm  = N/fs

        sig = A*np.exp(2j*np.pi*( f0*t + incF*np.square(t)/2/Tm )+ phi0*1j ) 
        
        sig = np.asarray(sig,dtype= np.complex)
        
        if (callback in  ['None', None]):
              return sig

        elif type(callback ) is not list:
            callback = list([callback])

            for callback_i in callback:
                sig = callback_i(sig)
           
            return sig   