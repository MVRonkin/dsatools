import numpy as np
import scipy

from .. import _frequency
from . import _fullphase

__all__ = ['full_unwraped_phase', 
           'maxcor_unwraped']

_MAX_TAU_SCALE_ = 3.33
#--------------------------------------------------------------  
def tau_unwraped_phase(s1, s2=None, f0=1, delta_f=1, Tm=None, 
                       fs=None, w_on=True, max_tau_scale=_MAX_TAU_SCALE_):
    '''
    Time delay difference estimations
      between two complex-valued  beat signals, obtained by frequency 
      modulated continium waves signals (FMCW). Method is based on the 
      phase-to-time approximation of the beat signals by the weigthed 
      least-square method, with weigths equal to its modules.
          
    Parameters
    ----------------
    * s1: 1d ndarray (complex),
        is the input signal.
    * s2: 1d ndarray (complex),
        if s2 is not none, than the dealy of 
        the conjugated product of s1,s2 will be estmated. 
    * f0: float,
        is the initial frequency.
    * delta_f: float,
        is the frequency band.    
    * T_m: float,
        is the period of modulation
        (T_m = x.shape[0]/fs if None).
    * fs: float,
        sampling frequency 
        (if None fs = x.shape[0]).
     * w_on: bool,
         if False, than |s(n)| will be set ones.
    * max_tau_scale: float,
        regularization parameter.
    
    Returns
    ------------------
    * delta_tau: float,
        estimated time delay difference.
    
    Parameters
    ----------------
    * s1: is the input signal.
    * s2: if s2 is not none, than the dealy of 
        the conjugated product of s1,s2 will be estmated.    
    * f0: is the initial frequency
    * delta_f: is the frequency band.
    * T_m: is the period of modulation (T_m = x.shape[0]/fs if None).
    * fs: is the sampling frequency (if None fs = x.shape[0]).
    * w_on: if False, than |s(n)| will be set ones.
    * max_tau_scale: regularization parameter.

    References
    -------------------
    [1] Ronkin M.V., Kalmykov A.A., Zeyde K.M. 
        Novel FMCW-Interferometry Method Testing on an 
        Ultrasonic Clamp-on Flowmeter, IEEE Sensors Journal,  
        Vol 20 , Issue 11 , 2020  p. 6029 - 6037,  
        DOI: 10.1109/JSEN.2020.2972604.
    [2] Ronkin M.V., Kalmykov A.A. Investigation of the time 
        delay difference estimator for FMCW signals, 
        Proceedings of the 2nd International Workshop on 
        Radio Electronics & Information Technologies 
        (REIT 2 2017), 2017. p. 90-99, 
        http://ceur-ws.org/Vol-2005/paper-11.pdf.  
    '''     
    s1 = np.array(s1)
    N = s1.shape[0]
    
    if(fs is None): fs = N        
    if(Tm is None): Tm = N/fs
    
    if (s2 is None):
        return _full_unwraped_phase(s1, fs, f0, delta_f, Tm, w_on, max_tau_scale)
    
#     max_tau  = 1/(max_tau_scale*(f0+delta_f)/2) 
    
    s2 = np.array(s2)    
    if(s1.shape != s2.shape): raise ValueError('s1.shape != s2.shape')

    s = s1*np.conj(s2)  
    return  _full_unwraped_phase(s, fs, f0, delta_f, Tm, w_on, max_tau_scale)

    
#--------------------------------------------------------------  
def maxcor_unwraped(s1, s2=None, f0=1, delta_f=1, 
                    fs = None, max_tau_scale=_MAX_TAU_SCALE_ ):
    '''
    Time delay difference estimations
     between two continium complex-valued signals, including 
     beat signal, obtained by frequency modulated continium waves 
     signals (FMCW), and FMCW signals its self.
   
    Parameters
    ----------------
    * s1: 1d ndarray (complex),
        is the input signal.
    * s2: 1d ndarray (complex),
        if s2 is not none, than the dealy of 
        the conjugated product of s1,s2 will be estmated. 
    * f0: float,
        is the initial frequency.
    * delta_f: float,
        is the frequency band.    
    * fs: float,
        sampling frequency 
        (if None fs = x.shape[0]).     
    * max_tau_scale: float,
        regularization parameter.
    
    Returns
    ------------------
    * delta_tau: float,
        estimated time delay difference.      

    Notes
    -----------------------
    * method does not require frequrency deviation, if delta_f =0,
        than method gives dealy estimation by initial phase.        
    * The value of time dealys has restricted 
              unambiguous estimation range +-1/2\\pi(f_0+\\Delta f/2).
    * If s2 is None, than estimation will be perfermed for dealy of s1.
    * The estimator expression
      ..math::
      \\Delta\\tau = arccos(\\sum_n{rho0[n])/2\\pi(f_0+\\Delta f/2),
        
      where:
      * \\Delta\\tau is the estimated time delay difference;
      * rho0[n] = s1[n]*conj(s2[n])/sqrt(\\sum(s1^2*s2^2));
      * s1,s2 - beat signals time delay difference beyween 
          which is measured;
      * angle(s) is the angle (argument) of the complex value;
      * f_0 is the initial frequency of FMCW signal;
      * \\Delta f is the frequency band (frequency deviation) of 
         the corresponing FMCW signal (from f_0 to f_0+\\Delta f);
      * T_m is the period of modulation.

    References
    ----------------------
    [1] Liao Y, Zhao B.,
        Phase-shift correlation method fot accurate phase difference
        estimation in range fider, Application optic, 
        v.54 # 11 p. 3470-3477.
    [2] Bjorklund S., A survey and comparison of time-delay estimation 
        methods in linear systems.— UniTryck: Linkoping, Sweden, 
        2003. —169 p.
    '''    
    s1 = np.array(s1)
    N = s1.shape[0]
    
    if(fs is None): fs = N 
    
    if (s2 is None):
        return _maxcor_unwraped(s1, f0, delta_f, fs, max_tau_scale)
    
#     max_tau  = 1/(max_tau_scale*(f0+delta_f)/2) 
    
    s2 = np.array(s2)    
    if(s1.shape != s2.shape): 
        raise ValueError('s1.shape != s2.shape')

    s = s1*np.conj(s2)  
    return  _maxcor_unwraped(s, f0, delta_f, fs, max_tau_scale)
   
    
#--------------------------------------------------------------                   
def _full_unwraped_phase(s, fs, f0, delta_f, Tm, w_on, max_tau_scale):

    max_tau  = 1/(max_tau_scale*(f0+delta_f)/2)

    f1       = _frequency.fitz_r(s,fs)

    t_coarse = f1*Tm/delta_f
    
    t_int    = np.fix(t_coarse/max_tau)*max_tau

    n = np.arange(s.shape[0])
    s_ref = np.exp(2j*np.pi*((delta_f/Tm*n/fs+f0)*t_int))

    tst = _fullphase.tau_fullphase(s, 
                                   s_ref, 
                                   f0, 
                                   delta_f, 
                                   Tm, 
                                   fs,                                    
                                   w_on = w_on) 

    tst = tst + t_int

    return tst



    
#--------------------------------------------------------------                   
def _maxcor_unwraped(s, f0, delta_f, fs, max_tau_scale):
    Tm = s.shape[0]/fs
   
    max_tau  = 1/(max_tau_scale*(f0+delta_f)/2)

    f1       = _frequency.fitz_r(s,fs)

    t_coarse = f1*Tm/delta_f
    
    t_int    = np.fix(t_coarse/max_tau)*max_tau

    n = np.arange(s.shape[0])
    s_ref = np.exp(2j*np.pi*(delta_f/Tm*n/fs+f0) *t_int)

    tst = _fullphase.maxcor(s, s_ref, f0, delta_f) 

    tst = tst + t_int

    return tst