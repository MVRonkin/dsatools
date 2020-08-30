import numpy as np
import scipy

from .. import _frequency
from . import _fullcorrphase

__all__ = ['tau_unwraped_corrphase']

_MAX_TAU_SCALE_ = 3.33
#--------------------------------------------------------------  
def tau_unwraped_corrphase(s1,s2=None,
                           f0=1,delta_f=1,Tm=None,fs=None, 
                           mode='R12', max_tau_scale=_MAX_TAU_SCALE_):
    '''
    Time delay difference estimations
      between two complex-valued  beat signals, obtained by 
      frequency modulated continium waves signals (FMCW). 
      Method is based on the phase-to-time approximation of 
      the beat signals by the weigthed 
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
    * mode: string,
        cross-correlation mode
        {'R12','R21','Rfb','Rfb12', 'Rfb21'}.
    * max_tau_scale: float,
        regularization parameter.
    
    Returns
    ------------------
    * delta_tau: float,
        estimated time delay difference.
    
    Notes
    -----------------
    * If fs=x.shape[0], 
        than delay will be calculated in points units.
    * The method is based on the supposition, that full phase 
        (frequency*time+inital_phase)
        depends only on delay (or delay difference).
    * The value of time dealys has restricted 
        unambiguous estimation range +-1/2\\pi(f_0+\\Delta f/2).
    * Basic estimator:
      ..math::
      Delta_tau = sum_n{W[n]|R[n]|angle(R[n])}/sum{W^2[n]|R[n]|},
       
      where:
      * Delta_tau is the estimated time delay difference;
      * R[n] is the correlation function;
      * angle(R) is the angle (argument) of complex-valued signals;
      * W[n] is the weigth function;
        there are following types of correlation and 
        corresponing weigths are aviliable:
          * R12 = R(s1,s2)R^*(s1,s1);    
          * W = 2\\pi[(\\Delta f * /(T_m*fs))*(N-1+n)/2+f_0];
            
          * R21 = R(s1,s2)R^*(s2,s2);    
          * W = 2\\pi[(\\Delta f * /(T_m*fs))*(N-1-n)/2+f_0];
            
          * Rfb = R(s1,s2)R^*(s2,s1);    
          * W = 2\\pi[(\\Delta f * /(T_m*fs))*(N-1)+2*f_0];
          
          * Rfb12 = (R(s2,s1)R^*(s2,s2))*(R(s1,s2)R^*(s1,s1))^*;    
          * W = 2\\pi[(\\Delta f * /(T_m*fs))*(N-1+n)+2*f_0];
            
          * Rfb21 = (R(s2,s1)R^*(s1,s1))*(R(s1,s2)R^*(s2,s2))^*;    
          * W = 2\\pi[(\\Delta f * /(T_m*fs))*(N-1-n)+2*f_0];            
            
          where:
          * f_0 is the initial frequency of FMCW signal;
          * \\Delta f is the frequency band (frequency deviation) of 
           the corresponing FMCW signal (from f_0 to f_0+\\Delta f);
          * T_m is the period of modulation.
  
    * The estimator is based on the following beat signal model:
      ..math::
      s[n] = a[n](exp{2j\\piW[n]\\tau})+s_par[n]+noises[n], 
       a[n]>|s_par[n]|  (high signal-to-interferences ratio),
       a[n]>|noises[n]| (high signal-to-noises ratio),
        
       where:
       * a[n] is the amplitude of valuable signal;
       * s_par[n] are the influence of the interference signals;
       * noises are the white gaussian noies.

    References
    ----------------------
    [1] Ronkin M.V., Kalmykov A.A., Zeyde K.M. 
        Novel FMCW-Interferometry Method Testing on 
        an Ultrasonic Clamp-on Flowmeter, IEEE Sensors Journal,  
        Vol 20 , Issue 11 , 2020  pp. 6029 - 6037,  
        DOI: 10.1109/JSEN.2020.2972604.
    [2] Ronkin M.V., Kalmykov A.A.,
        A FMCW - Interferometry approach for ultrasonic flow meters,
        2018 Ural Symposium on Biomedical Engineering, 
        Radioelectronics and Information Technology (USBEREIT), 
        2018. p. 237 – 240.  DOI: 10.1109/USBEREIT.2018.8384593.  
    [3] Ronkin M.V., Kalmykov A.A. Investigation of the time delay 
        difference estimator for FMCW signals, 
        Proceedings of the 2nd International Workshop on 
        Radio Electronics & Information Technologies 
        (REIT 2 2017), 2017. pp.90-99, 
        http://ceur-ws.org/Vol-2005/paper-11.pdf.  
    
    '''   
    s1 = np.array(s1)
    N = s1.shape[0]
    
    if(fs is None): fs = N        
    if(Tm is None): Tm = N/fs
    
    if (s2 is None):
        return _full_unwraped_corrphase(s1, fs, f0, delta_f, Tm, mode, max_tau_scale)
    
#     max_tau  = 1/(max_tau_scale*(f0+delta_f)/2) 
    
    s2 = np.array(s2)    
    if(s1.shape != s2.shape): raise ValueError('s1.shape != s2.shape')

    s = s1*np.conj(s2)  
    return  _full_unwraped_corrphase(s, fs, f0, delta_f, Tm, mode, max_tau_scale)

#--------------------------------------------------------------                   
def _full_unwraped_corrphase(s, fs, f0, delta_f, Tm, mode, max_tau_scale):

    max_tau  = 1/(max_tau_scale*(f0+delta_f)/2)

    f1       = _frequency.fitz_r(s,fs)

    t_coarse = f1*Tm/delta_f
    
    t_int    = np.fix(t_coarse/max_tau)*max_tau

    n = np.arange(s.shape[0])
    s_ref = np.exp(2j*np.pi*((delta_f/Tm*n/fs+f0 ) *t_int))

    tst = _fullcorrphase.tau_fullcorrphase(s, 
                                           s_ref, 
                                           f0, 
                                           delta_f, 
                                           Tm,
                                           fs,
                                           mode = mode) 

    tst = tst + t_int

    return tst

