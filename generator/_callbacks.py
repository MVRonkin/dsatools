import numpy as np
import scipy

import sys
# sys.path.append(r'P:/PyProjects/Libraries/')
from .. import utilits as ut


from ._simsig_tools import _check_list,_rand_uniform

__all__ = ['probe_modulation',
           'harmonic_modulataion',
           'pulse_modulataion',
           'pulse_shift_modulataion',
           'shift_modulataion',
          ]

#--------------------------------------------------------------------------------- 
def probe_modulation(callback, length=512):
    ''' 
     Probe (plot in time and frequency domain) callback responce.
        
     Notes
     ------------
     * Callback should be in the simsignal class format.
     * See simsignal.
        
     Parameters
     --------------
     * length: int,
        length of output should be equal to required.
     * callback: function (callable),
        function in the format f(x).
    
    Example
    ---------
        import dsatools.generator 
        from dsatools.generator import callbacks
        cb1 = callbacks.harmonic_modulataion(amp_am=0.1,freq_am=0.5,phase_shift=0)
        callbacks.probe_modulation(cb1,512)
        cb2 = callbacks.pulse_modulataion(200,400)
        callbacks.probe_modulation(cb2,512)
    '''
    ones = np.ones(length)
    if (callback is not None):
        ones = callback(ones)
    print(ones.shape)    
    ut.probe (ones)
    
#---------------------------------------------------------------------------------    
def harmonic_modulataion(amp_am = 0.1, freq_am = 0.1, phase_shift = 0 ):
    ''' 
    Harmonic amplitude modulation

    Parameters
    -----------
    * amp_am: float,
        amplitude of modulation.
    * freq_am: float,
        frequency of modulation.
    * phase_shift: float,
        phaseshift of modulation.
        
    Returns
    ------------
    * lambda: callable,
        function for callback in simsignal class format.
    
    Notes
    -----------
    * For test modulation use probe_modulation.
    * Harmonic amplitude modulation in the form:
      ..math::
      (1+a*cos(2pi*f*n/N + \phi_0) )*x/2,
      where: 
      * a is the amplitude of modulation;
      * f is the frequency of modulation;
      * phi_0 frequency inital phase of modulation;
      * x is the input signal.    
        
    Example
    ----------------
        import dsatools.generator 
        from dsatools.generator import callbacks
        cb1 = callbacks.harmonic_modulataion(amp_am=0.1,freq_am=0.5,phase_shift=0)
        callbacks.probe_modulation(cb1,512)
        cb2 = callbacks.pulse_modulataion(200,400)
        callbacks.probe_modulation(cb2,512)

    '''
    return lambda x: (1+amp_am*np.cos(
        2*np.pi*freq_am*np.arange(x.shape[0])/x.shape[0] + phase_shift))*x/2
#------------------------------------------------------------------------

def pulse_modulataion(start_position = 0, stop_poistion = None ):    
    '''
    Pulse modulation function, 
    for return a part of harmoinc signal (i.e. pulse)

    Parameters
    ----------
    * start_position: int,
        first non-zero point of pulse.
    * stop_poistion: int,
        last non-zero point of pulse +1.
    
    Returns
    ----------
    * lambda: callable,
        function for callback in simsignal class format.
        
    Notes
    -----------
    * for test modulation use probe_modulation.
    * Harmoinc signal (i.e. pulse):    
      x = (0,..,0,x[start_postion:stop_poistion],0,..0).
    
    Example
    -----------
        probe_modulation(512, pulse_modulataion(
                start_position = 100, stop_poistion =146))
    '''
    start_position = int(start_position)   
    
    if (start_position <0):
        raise ValueError('start_position<0')    
    
    if(stop_poistion is None):       
        return lambda x: np.append(np.zeros(start_position),x[start_position:])
    
    stop_poistion  = int(stop_poistion)
    
    if(stop_poistion <0):       
        return lambda x: np.concatenate((np.zeros(start_position),
                                         x[start_position:stop_poistion],np.zeros(-stop_poistion)))     
    
    if(start_position>0 and stop_poistion>0 and stop_poistion<start_position):
        raise ValueError('stop_poistion<start_position')
    
    return lambda x: np.concatenate((np.zeros(start_position),
                                     x[start_position:stop_poistion],np.zeros(x.shape[0]-stop_poistion)))         
#------------------------------------------------------------------------

def pulse_shift_modulataion(start_position = 0, stop_poistion = None, shift=0 ):
    '''
    Pulse shifted modulation function, 
    for return a part of harmoinc signal (i.e. pulse).
    
    Parameters
    ----------
    * start_position: int,
        first non-zero point of pulse.
    * stop_poistion: int,
        last non-zero point of pulse +1.
    * shift: int,
        shift pulse if it necessary.
        
    Returns
    ----------
    * lambda: callable,
        function for callback in simsignal class format.
        
    Notes
    -------------
    * for test modulation use probe_modulation.
    * Harmoinc signal (i.e. pulse).    
        x = (0,..,0,x[start_postion+shift:stop_poistion+shift],0,..0)    
       
    Example
    ---------
        probe_modulation(512, pulse_shift_modulataion(
            start_position = 50, stop_poistion =-200, shift=-50 ))
        
    '''    
    start_position = start_position+shift
    
    if (start_position<0):
        start_position = 0
        
    if(stop_poistion is None and shift < 0):
        stop_poistion = shift
    
    if(stop_poistion is not None):
        if(stop_poistion >0):
            stop_poistion = stop_poistion+shift

        if(stop_poistion <0):
            stop_poistion = stop_poistion+shift
    
    return pulse_modulataion(start_position, stop_poistion )
#------------------------------------------------------------------------
def shift_modulataion(shift=0, extened=False ):
    '''
    Shift of signal,
        x = (0,..,0,x[:x.shape[0]-shift])
    
    if self-similarity:
        x = (x[x.shape[0]-shift:],x[:x.shape[0]-shift])
        
    Notes
    ----------
    * if shift <0 shift will be reversed:
      x = (x[shift:],0,...,0)
      or
      x = (x[shift:],x[:shift])
        
    Examples
    -----------
        probe_modulation(512,shift_modulataion(shift=50 ))
        probe_modulation(512,shift_modulataion(shift=50, extened = True ))
        probe_modulation(512,shift_modulataion(shift=-50 ))
        probe_modulation(512,shift_modulataion(shift=-50, extened = True ))
             
    '''
    if(shift<0):
        shift = -shift
        if (extened):
            return lambda x: np.append(x[shift:],x[:shift]) 
        else:
            return lambda x: np.append(x[shift:],np.zeros(shift))              
        
    if(shift>=0):
        if (extened):
            return lambda x: np.append(x[x.shape[0]-shift:],x[:x.shape[0]-shift])  
        
        else:
            return lambda x: np.append(np.zeros(shift),x[:x.shape[0]-shift])
