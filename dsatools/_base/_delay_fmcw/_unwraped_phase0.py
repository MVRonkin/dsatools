import numpy as np
import scipy
# from ... import utilits as ut

from .. import _frequency

from .. import _phase_init

__all__ = []

_MAX_TAU_SCALE_ = 12.33
#--------------------------------------------------------------  
def ls_unwraped(s1, s2=None, f0=1, delta_f=1, fs = None, w_on = True, max_tau_scale=_MAX_TAU_SCALE_ ):
    '''
    
    '''    
    s1 = np.array(s1)
    N = s1.shape[0]
    
    if(fs is None): fs = N 
    
    if (s2 is None):
        return _ls_unwraped(s1, fs, f0, delta_f, w_on, max_tau_scale)
    
#     max_tau  = 1/(max_tau_scale*(f0+delta_f)/2) 
    
    s2 = np.array(s2)    
    if(s1.shape != s2.shape): raise ValueError('s1.shape != s2.shape')

    s = s1*np.conj(s2)  
    return  _ls_unwraped(s, fs, f0, delta_f, w_on, max_tau_scale)
   
    
#--------------------------------------------------------------                   
def _ls_unwraped(s, fs, f0, delta_f, w_on, max_tau_scale):
    
    Tm = s.shape[0]/fs
    
    max_tau  = 1/(max_tau_scale*(f0+delta_f)/2)

    f1       = _frequency.fitz_r(s,fs)

    t_coarse = f1*Tm/delta_f
    
    t_int    = np.fix(t_coarse/max_tau)*max_tau

    n = np.arange(s.shape[0])
    
    s_ref = np.exp(2j*np.pi*((delta_f/Tm*n/fs+f0 ) *t_int))
    
    if(w_on):
        tst = _phase0.wls_phase(s, s_ref)/(2*np.pi*f0)
    
    else:
        tst = _phase0.ls_phase(s, s_ref)/(2*np.pi*f0)

    tst = tst + t_int

    return tst


