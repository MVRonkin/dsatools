import numpy as np
import scipy


__all__ = ['wls_phase0','ls_phase0','Tretter_phi0','mlfft_phi0']


def __check_input__(s1,s2):
    s1 = np.array(s1)
    N = s1.shape[0]
    if(s2 is None):
        return s1, N
    
    s2 = np.array(s2)
    if(s1.shape !=s2.shape):
        raise ValueError('s1.shape !=s2.shape')
    
    s = s1*np.conj(s2)
    return s, N

#--------------------------------------------------------------  
def wls_phase(s1,s2=None):
    '''
    Fu Н., Kam P.Y. initial phase estimator. 
      Same as to weigthed least square approximation of singals 
      conjugated product phase,with weigth equal to its module.

    Parameters
    ---------------------
    * s1,s2: 1d ndarray (complex),
        are the input 1d ndarrays (input signals).
        if s2 is None, initial phase of s1 will 
        be measured only.
    
    Returns
    ---------------------
    * phi0: float,
        estimated initial phase.   
    
    
    Notes
    ----------------------------
    * See also ls_phase.
    * phi0 has restricted unambiguous estimation range [-\\pi;\\pi].
    * The estiamtor:
      ..math::
      phi0 = [1/\\Phi][\\sum (|s|n^2) * sum (|s|arg(s)) 
                                - sum(n|s|)sum(n|s|arg(s))],
      where:        
      * phi0 is the estimated initial phase (in radians);         
      * s = s(n) is the signal to estimate;
      * Psi = sum(|s|)*sum(n^2|s|)-[sum(n|s|)]^2;
      * arg(s(n)) = angle s(n);
      * n = 0,...,N-1, (N is the input length). 
    
    Refernces
    --------------------------------
    [1] Fu Н., Kam P.Y. 
        Linear estimation of the frequency and phase of a noisy sinusoid,
        IEEE Trans. on Inform. Theory. 
        2003. V.31. p. 832 - 835.
    [2] Fu Р. Kam P.Y.,
        ML Estimation of the Frequency and Phase in Noise, 
        Proceeding of IEEE Globecom 2006. 
        2006. p. 1-5.          
    
    '''    
    s, N =  __check_input__(s1,s2)

    n      = np.arange(N)    
    ph     = np.angle(s)
    n2     = np.square(n)
    
    W      = np.abs(s)
    phires = (np.sum(W*n2)*np.sum(W*ph)-np.sum(W*n)*np.sum(W*ph*n))/\
             (np.sum(W)*np.sum(W*n2)-np.sum(W*n)**2)

    return  - phires

#--------------------------------------------------------------  
def ls_phase(s1,s2=None):
    '''
    Least-square initial phase estimator. 
      Based on the least square approximation of singals 
      conjugated product phase.

    Parameters
    ---------------------
    * s1,s2: 1d ndarray (complex),
        are the input 1d ndarrays (input signals).
        if s2 is None, initial phase of s1 will 
        be measured only.
    
    Returns
    ---------------------
    * phi0: float,
        estimated initial phase. 
    
    Notes
    ---------------------
    * See also wls_phase.
    * phi0 has restricted unambiguous estimation range [-\\pi;\\pi].
    * The estiamtor:
      ..math::
      phi0 = [1/Phi][sum (n^2)*sum (arg(s)) 
                                - sum (n)sum(narg(s))],
     
      where:        
      * phi0 is the estimated initial phase (in radians);         
      * Psi = [N(N+1)/2]^2 - N(N(N+1)(N+2))/6;
      * arg(s(n)) = angle s(n);
      * n = 0,...,N-1, (N is the input length).

    Refernces
    -------------------------
    [1] Tretter S. A., 
        Estimating the frequency of a noisy sinusoid by linear regression, 
        IEEE Trans. Inform. Theory..
        1985. v. IT-3 1. p. 832-835.
    
    
    
    '''    
    s, N =  __check_input__(s1,s2)
    
    n      = np.arange(N)
    ph     = np.angle(s)
    n2     = n**2
    
    wnum   = np.sum(n2)*np.sum(ph)-np.sum(n)*np.sum(ph*n)
    wdenum = (N*(N+1)/2)**2 - N*(N*(N+1)*(N+2))/6
    phires = wnum/wdenum
    
    return  - phires



#-------------------------------------------------------------- 
def tretter_phase(s1,s2=None):
    '''
    Initial phase estimation 
        base on Tretter signal model.

    Parameters
    ---------------------
    * s1,s2: 1d ndarray (complex),
        are the input 1d ndarrays (input signals).
        if s2 is None, initial phase of s1 will 
        be measured only.
    
    Returns
    ---------------------
    * phi0: float,
        estimated initial phase.   
    
    Notes
    ---------------------------
    * Function FOR TEST ONLY - closed form has been derived 
        with some simplifications which lead to some bias
        in estimation.
    * phi0 has restricted unambiguous estimation range [-pi;pi].
    * The estimator:
      ..math::
      phi0 = 6sum(n-2N/3)arg(s)/N^2 
      where:
      * phi0 is the estimated initial phase (in radians);         
      * arg(s(n)) = angle s(n);
      * n = 0,...,N-1, (N is the input length).

    References
    ------------------------
    [1] Tretter S. A., 
        Estimating the frequency of 
        a noisy sinusoid by linear regression, 
        IEEE Trans. Inform. Theory..
        1985. v. IT-3 1. p. 832-835.
    '''
    s, N =  __check_input__(s1,s2)
    n      = np.arange(N)
    w      = (n-2*N/3)/N**2
    phi0   = 6*np.sum(w*np.angle(s))
    return   phi0  

#-------------------------------------------------------------- 
def maxcor_phase(s1,s2, normalize = False):
    '''
    Initial phase estimation 
        base on correlation coefficient (i.e. angle) 
        between signals model.

    Parameters
    ---------------------
    * s1,s2: 1d ndarray (complex),
        are the input 1d ndarrays (input signals).
        if s2 is None, initial phase of s1 will 
        be measured only.
    * normalize: bool,
        if True, than normalized values
        will be taken.
    
    Returns
    --------
    * phi0: float,
        estimated initial phase.    
    
    Notes
    ------
    * If real values: phase = arccos(correlation_cof)
        else: phase = arctan2( Im{correlation_cof}, Re{correlation_cof})
    * If normalize:
            correlation_cof = sum(x*conj(y))/sqrt(sum(x*conj(x))*sum(y*conj(y))),
        else: correlation_cof = sum(x*conj(y)).
    * Normalization is necessary if inputs are real valued signals.
    Returns
    ---------------------
    * phi0: float,
        estimated initial phase.   
    '''
    s, N =  __check_input__(s1,s2)
    out = np.sum(s2*np.conj(s1))
    if (normalize):
        out /= np.sqrt(np.abs(np.sum(np.square(s1))*np.sum(np.square(s2))))
    
    # if any iscomplex
    if s1.dtype in [np.complex,complex,np.complex128,np.complex64] or \
       s2.dtype in [np.complex,complex,np.complex128,np.complex64]:
        phi0 = np.angle(out)
    
    else:
        phi0 = np.arccos(out)

    return   phi0 

#-------------------------------------------------------------- 
def mlfft_phase(s1,s2=None,Nfft = None):
    '''
    FUNCTION DOES NOT WORK CORRECTLY
    
    Maximum-likelihood estimator of initial phase,
      based on the maximum searching of the signal spectrum,
      obtained by fast Fourier transform with zero-padding.
 
    Parameters
    -----------------------------
    * s1: is the input 1d ndarrays (input signals);  
    * s2: is the input 1d ndarrays (input signals), could be None;  
    * Nfft: length of signal spectrum (with zerro-padding), 
                                            Nfft = s1.size if None.
    
    Returns
    ------------------------
    * phi0 - estimated initial phase.
    
        
    Notes
    -------------------
    * Function FOR TEST.
        
    Referenes
    ------------------------
    [1] Rife D. and Boorstyn R., 
        Single-tone parameter estimation from discrete-time observations, 
        IEEE Transactions on Information Theory, 
        vol. 20, № 5, 1974, p. 591–598.
        
    '''
    s1 = np.asarray(s1)
    N  = s1.shape[0] 
    
    if(not Nfft): Nfft = N
#     if(not fs):   fs = Nfft    
        
    LimOfFind = int(Nfft/2)
    
    S  = np.fft.fft(s1,Nfft) 
    pp = np.flatnonzero(np.abs(S)==np.max(np.abs(S)))[0]   
    phi0_s1 =  np.angle(S[pp])    
    
    if(s2 is None):    
        return   phi0_s1
    else:
        s2 = np.asarray(s2)
        S  = np.fft.fft(s2,Nfft) 
        pp = np.flatnonzero(np.abs(S)==np.max(np.abs(S)))[0]   
        phi0_s2 =  np.angle(S[pp])         
        return phi0_s2-phi0_s1
        