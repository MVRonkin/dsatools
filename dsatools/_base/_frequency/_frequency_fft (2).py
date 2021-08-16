import numpy as np


__all__ = ['mlfft','barycenter_fft', 'barycenter_general_gauss_fft']

#--------------------------------------------------------------
def _check_input(s,fs=None):
    s = np.array(s)
    N = s.shape[0]
    
    if(fs is None):
        fs = N
    return s,N,fs

#---------------------------------------------------------------- 
def ml_fft(s,fs = None,n_fft = None):
    '''
    Maximum-likelihood estimator of frequency,
      based on the maximum searching of the signal spectrum,
      obtained by fast Fourier transform with zero-padding.

    Parameters
    -------------------
    * s: 1d ndarray,
        input signal.
    * fs: float,
        sampling frequency 
        (fs = Nfft, if None).
    * n_fft: int or None,
        length of signal spectrum (with zerro-padding), 
        n_fft = s.size if None.

    Returns
    ------------------
    * f: float,
         estimated frequency.  
    
     Note
    ------------------
    * if fs = N, then f will be measured in points.
        
    Referenes
    ------------------
    [1] Rife D. and Boorstyn R., 
        Single-tone parameter estimation from 
        discrete-time observations, 
        IEEE Transactions on Information Theory, 
        vol. 20, № 5, 1974, p. 591–598.    
    [2] Fowler M.L., 
        Phase-based frequency estimation: a review, 
        Digital Signal Processing. 
        2012. v. 12. p. 590–615.
        
    '''
    s,N,fs = _check_input(s,fs)
    
    if(not n_fft): n_fft = N
        
    if(not fs):   fs = n_fft        
        
    lim = int(n_fft/2)
    
    S      = np.abs(np.fft.fft(s,n_fft))  
    
    S      = np.hstack((np.zeros(1),S[1:lim]))
    
    pp     = np.flatnonzero(S==max(S))[0]   
    
    f_res  = fs*(pp)/n_fft     
    
    return   f_res

#-------------------------------------------------
def barycenter_fft(x,band=None,fs=None, n_fft = None):
    '''
    Barycenter (centre of mass) 
        based estimator in the spectrum domain.
    
    Parameters
    ------------------
    * x: 1d ndarray.
    * band: [int,int]
        value or [low,high] values
        - bands of the pass-band width,
        if band is 1 value, 
        than band will be [band,band].
    * fs: float,
        sampling frequency.
    * n_fft: int or None,
        length of samples size for spectrum obtaning,
        if None n_fft = x.size.
    
    Returns
    ----------------
    * f_est: float,
        estimated frequency.
    
    Notes
    ---------------

    '''
    
    x,N,fs = _check_input(x,fs)

    if n_fft is None: n_fft = N
    
    #TODO: fftshift for near-zero values?
    sp = np.fft.fft(x, n_fft)[:n_fft//2]
    sp = (sp*np.conj(sp)).real    

    if band is not None:
        band = np.asarray(band)        
        if band.size==1: band = np.array([band,band])
        pband = np.asarray(n_fft*band/fs,dtype = np.int)
        
        p_max = np.flatnonzero(sp==np.max(sp))[0]
        
        w_band = [-pband[0]+p_max, pband[1]+p_max]

        if w_band[0]>w_band[1]: w_band = [w_band[1],w_band[0]]            
        if w_band[0]<0:w_band[0]=0            
        if w_band[1]>n_fft//2: w_band[1] = n_fft//2
    
    else:
        w_band = np.array([0,n_fft//2],dtype = np.int)

    sp = sp[w_band[0] : w_band[1] ]

    n = fs*np.arange(w_band[0], w_band[1])/n_fft

    return np.sum(n*sp)/np.sum(sp)

#-------------------------------------------------
def barycenter_general_gauss_fft(x, sigma = 1/2, 
                                 degree = 2, fs=None, n_fft = None):
    '''
    Barycenter (centre of mass) based
        an estimator in spectrum domain 
        based on the general_gauss windowing.
    
    Parameters
    ------------------
    * x: 1d ndarray.
    * sigma: float,
        than higher it value, than wider pass-band.
    * degree: flaot,
        than higher it value, than closer to the square window.
    * fs: float,
        sampling frequency.
    * n_fft: int or None,
        length of samples size for spectrum obtaning,
        if None n_fft = x.size.
    
    Returns
    ----------------
    * f_est: float,
        estimated frequency
  
    ''' 
    
    x,N,fs = _check_input(x,fs)
    if n_fft is None: n_fft = N
    
    #TODO: fftshift for near-zero values?
    sp = np.fft.fft(x, n_fft)[:n_fft//2]
    sp = (sp*np.conj(sp)).real    
    
    p_max = np.flatnonzero(sp==np.max(sp))[0]
    
    w=_general_gauss(n_fft//2, n0 =p_max, a=sigma, p=degree)

    sp = sp*w

    n = fs*np.arange(n_fft//2)/n_fft

    return np.sum(n*sp)/np.sum(sp)


#-------------------------------------------------
def _general_gauss(N, n0 =None, a=0.5, p=2):
    out=np.zeros(N) 
    
    p = 2*int(p//2)
    
    if n0 is None: n0 = N//2
    n = np.arange(N)
    out = np.exp(-np.power((n-n0)/(a*N/2),p))
    return out
