import numpy as np
import scipy

def hht(components):
    '''
    Hilbert-Huang Transform (HHT).
    
    Parameters
    ------------
    * components: 2d ndarray,
        with size (number of components x sample size of each one),
        components are the same as intrinsic mode function(IMF)
        for Emperical Mode Decomposition (signal components).

    
    Returns
    ----------
    * envelopes: 2d ndarray,
        envelopes for each IMF.
    * instantfreqs: 2d ndarray,
        instant frequencies for each IMF.
    
    Notes
    -------
    * Function for work with IMF decompositions,
       such as VMF, EMD, HVD, EWT and etc., see
       decimpositions.

    References
    --------------
    [1] N. E. Huang et al., 
        "The empirical mode decomposition and the Hilbert 
       spectrum for nonlinear and non-stationary time series analysis", 
       Proc. R. Soc. Lond. A, Math. Phys. Sci., 
       vol. 454, no. 1971, 903–995, (1998).
    [2] N. E. Huang, 
        "Hilbert-Huang transform and its applications", 
        vol. 16. World Scientific, 2014.      
    
    '''
    components = np.asarray(components)
    
    hilbert_flag = False
    if components.dtype not in \
        [complex,np.complex,np.complex64,
		np.complex128,np.complex_]:
         hilbert_flag = True
    
    envelopes    = np.zeros_like(components)
    instantfreqs = np.zeros_like(components)
    
    for i,component in enumerate(components):
        
        if hilbert_flag:
            component = scipy.signal.hilbert(component) 
        envelopes[i,:] = np.abs(component)
        
        #TODO: optimize, uzing explicity unwrap rutine
        instantfreqs[i,:] = ut.diff(np.unwrap(np.angle(component)))/2/np.pi
    return envelopes, instantfreqs