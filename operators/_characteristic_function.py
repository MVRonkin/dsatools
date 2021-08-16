
import numpy as np


__EPSILON__ = 1e-8
def ecf(x,y=None,fs=1,take_mean = False):
    '''
    Empirical Characteristic Function (ECF).
    
    Parameters
    ----------
    * x,y 1d ndarrays,
        input samples.
    * fs: float,
        sampling frquency.
    * take_mean:
        if true mean value will be sustructed.
    
    Returns
    ----------
    * ecf: empirical characteristic function.
    
    Notes
    -------
    ECF calculated as:
    * out[t] = sum_i(exp(jt*x[i]))
       if y is not None: 
           out[t] = sum_i(exp(jt*(x[i]-y[i])))
       The analogue of 
    * Ecf is the probability-generating 
        function but for emeprical case.
               
    '''
    x = np.asarray(x)
    if y is None: y = x
    else: 
        y = np.asarray(y)
        if (y.shape != x.shape): raise ValueError('y.shape ! = x.shape')    
            
    t = np.arange(x.shape[0])/fs
  
    out = [np.sum(np.exp(-1j*ti*np.abs(x-y))) for ti in t]
    return  np.asarray(out,dtype = np.complex)

    

    
    
    
    
    
    
    
    
