import numpy as _np

from ._base import _subspace 
from ._base import _arma 
from ._base._classic_psd import(capone,
                                slepian, 
                                periodogram,
                                correlogram, 
                                welch, 
                                bartlett, 
                                blackman_tukey,
                                daniell,
                                kernel_periodogram) 

__all__ = ['music',
           'ev',
           'phd',
           'phd_cor',
           'minvariance',
           'kernel_noisespace',
           'kernel_signalspace',
           'capone',
           'slepian', 
           'periodogram',
           'correlogram', 
           'welch', 
           'bartlett', 
           'blackman_tukey',
           'daniell',
           'kernel_periodogram',
           'pminnorm',
           'psd_kernel_minnorm',
           'phoyw']

#----------------------------------------------
def music(x, order, mode='full', lags=None, n_psd=None):    
    '''  
    Estimation the pseudo-spectrum based on the
    MUltiple SIgnal Classification (MUSIC) 
    algorithm, determinated as noise 
      subspace signal part.

    Parameters
    --------------
    * x: 1d ndarray,
        input signal.
    * order: int,
        order of the model
        (number of valuable components, 
        size of signal subspace). 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    * lags: int or None,
        number of lags in correlation function, 
        (x.shape[0]//2 by default).
    * n_psd: int or None,
        length of the desired pseudo-psectrum.
    
    Returns
    -------------
    * psd: 1d ndarray,
        estimated pseudo-spectrum.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
        
    Example
    -------------
   
    References
    ----------------
    [1a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page
    [2a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling
    
    '''    
    noise_space = _subspace.music(x=x, order=order, mode=mode, lags=lags) 
    psd = _subspace.subspace2psd(noise_space, n_psd )
    return psd

#-----------------------------------------
def ev(x, order, mode='full', lags=None,n_psd=None):    
    '''  
    Estimation the pseudo-spectrum based on the 
      Eigen Values (EV)algorithm model, 
      determinated as the normalized 
      noise-subspace signal part.

    Parameters
    --------------
    * x: 1d ndarray,
        input signal.
    * order: int,
        order of the model
        (number of valuable components, 
        size of signal subspace). 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    * lags: int or None,
        number of lags in correlation function, 
        (x.shape[0]//2 by default).
    * n_psd: int or None,
        length of the desired pseudo-psectrum.
        
    Returns
    -------------
    * psd: 1d ndarray,
        estimated pseudo-spectrum.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    
    Example
    -------------
        
    References
    ----------------
    [1a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page
    [2a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling
    
    '''     
    noise_space = _subspace.ev(x=x, order=order, mode=mode, lags=lags) 
    psd = _subspace.subspace2psd(noise_space, n_psd )
    return psd

#------------------------------------
def phd(x, order, mode='full',n_psd=None):    
    '''  
    Estimation the pseudo-spectrum based on the 
        Pisarenko Harmonic Decomposition (PHD) algorithm, 
        determined as the first noise-subspace vector.
        
    Parameters
    --------------
    * x: 1d ndarray,
        input signal.
    * order: int,
        order of the model
        (number of valuable components, size of signal subspace),
        and the number lags of covariance matrix - 1. 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    * n_psd: int or None,
        length of the desired pseudo-psectrum.
        
    Returns
    -------------
    * psd: 1d ndarray,
        estimated pseudo-spectrum.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    
    Example
    -------------
        
        
    References
    ----------------
    [1a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page
    [2a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling
    
    '''
    x = _np.asarray(x)
    if n_psd is None: n_psd = x.shape[0]
    noise_space = _subspace.pisarenko(x=x, order=order, mode=mode) 
    psd = _subspace.subspace2psd(noise_space, n_psd )
    return psd


#------------------------------------
def phd_cor(x, order, mode='full', cor_mode = 'straight',n_psd=None):    
    '''  
    Estimation the pseudo-spectrum based on the 
        Pisarenko Harmonic Decomposition (PHD) algorithm, 
        determined as the first noise-subspace vector,
        using additional calculation 
        of the correlation function. 

    Parameters
    --------------
    * x: 1d ndarray,
        input signal.
    * order: int,
        order of the model
        (number of valuable components, size of signal subspace),
        and the number lags of covariance matrix - 1. 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    * cor_mode: string,
        mode of additionally taken correlation function,
        cor_mode = {full,same,straight}.
    * n_psd: int or None,
        length of the desired pseudo-psectrum.
        
    Returns
    -------------
    * psd: 1d ndarray,
        estimated pseudo-spectrum.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    
    Example
    -------------
        
        
    References
    -----------------
    [1a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page
    [2a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling
    
    ''' 
    x = _np.asarray(x)
    if n_psd is None: n_psd = x.shape[0]
    noise_space = _subspace.pisarenko_cor(x=x, order=order, 
                                         mode=mode, cor_mode=cor_mode) 
    psd = _subspace.subspace2psd(noise_space, n_psd )
    return psd

#------------------------------------
def minvariance(x, order, mode='full',n_psd=None):    
    '''  
    Estimation of the pseudo-spectrum based on the 
        minimum variance (maximum likelyhood) algorithm,
        using normalized signal subspace. 

    Parameters
    --------------
    * x: 1d ndarray,
        input signal.
    * order: int,
        order of the model
        (number of valuable components, size of signal subspace),
        and the number lags of covariance matrix . 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    * n_psd: int or None,
        length of the desired pseudo-psectrum.
        
    Returns
    -------------
    * psd: 1d ndarray,
        estimated pseudo-spectrum.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    
    Example
    -------------
        
        
    References
    ----------------
    [1a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page
    [2a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling
    
    '''     
    x = _np.asarray(x)
    if n_psd is None: n_psd = x.shape[0]
    signal_space = _subspace.minvariance(x=x, order=order, mode=mode) 
    psd = _subspace.subspace2psd(signal_space, n_psd )
    return psd

#------------------------------------
def kernel_noisespace(x, order, mode='full', 
                      kernel = 'linear', 
                      kpar=1, lags=None, 
                      use_ev = False,n_psd=None):    
    '''  
    Estimation the noise subspace based  
        pseudo-spectrum of the kernel matrix taken
        for the input signal.

    Parameters
    --------------
    * x: 1d ndarray,
        input signal.
    * order: int,
        order of the model
        (number of valuable components, 
        size of signal subspace). 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    * kernel: string,
        kernel trick function,
        kernel type = {linear, poly, rbf, thin_plate}.
    * kpar: float,
        kernel parameter, depends on the kernel type.
    * lags: int or None,
        number of lags in correlation function, 
        (x.shape[0]//2 by default).
    * use_ev: bool,
        if True, than normalized space will be taken.
    * n_psd: int or None,
        length of the desired pseudo-psectrum.
        
    Returns
    -------------
    * psd: 1d ndarray,
        estimated pseudo-spectrum.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    
    Example
    -------------
        
    References
    ----------------
    [1a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page.
    [2a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling.
    
    '''   
    
    noise_space = _subspace.kernel_noisespace(x, 
                                             order, 
                                             mode=mode, 
                                             kernel=kernel,
                                             kpar=kpar, 
                                             use_ev=use_ev)
    
    psd = _subspace.subspace2psd(noise_space,n_psd=n_psd )

    return psd

#------------------------------------
def kernel_signalspace(x, order, mode='full', 
                       kernel = 'linear', kpar=1, 
                       lags=None, use_ev = False, n_psd=None):    
    '''  
    Estimation the signal subspace based  
        model of the kernel based signal.

    Parameters
    --------------
    * x: 1d ndarray,
        input signal.
    * order: int,
        order of the model
        (number of valuable components, 
        size of signal subspace). 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    * kernel: string,
        kernel trick function,
        kernel type = {linear, poly, rbf, thin_plate}.
    * kpar: float,
        kernel parameter, depends on the kernel type.
    * lags: int or None,
        number of lags in correlation function, 
        (x.shape[0]//2 by default).
    * use_ev: bool,
        if True, than normalized space will be taken.
    * n_psd: int or None,
        length of the desired pseudo-psectrum.
        
    Returns
    -------------
    * psd: 1d ndarray,
        estimated pseudo-spectrum.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    
    Example
    ------------

    References
    ----------------
    [1a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page.
    [2a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling.
    
    '''     
    signal_space = _subspace.kernel_signalspace(x, 
                                               order, 
                                               mode=mode, 
                                               kernel=kernel,
                                               kpar=kpar, 
                                               use_ev=use_ev)
    
    psd = _subspace.subspace2psd(signal_space,n_psd=n_psd )

    return psd
#---------------------------------------
def minnorm(x, order, mode='full',
            lags=None, signal_space = False, n_psd=None ):
    '''  
    Estimation autoregression model based pseudo-spectrum 
        based on the minimum-norm (maximum-entropy) algorithm.
    
    Parameters
    -----------
    * x:  1d ndarray. 
    * order: int.
        the autoregressive model (pole model) 
        order of the desired model. 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    * lags: int or None,
        number of lags in the correlation matrix 
        (lags =x.shape[0]//2 if None).
    * signal_space: bool,
        if True, than coefficients will be estimated
        in the signal subspace, else in noise one.
    * n_psd: int or None,
        length of the desired pseudo-psectrum.
        
    Returns
    -------------
    * psd: 1d ndarray,
        estimated pseudo-spectrum.
    
    Example
    ------------
    
    References
    ------------
    [1a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page
    [2a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling
        
    ''' 
    x = _np.asarray(x)
    if n_psd is None: n_psd = x.shape[0]
    
    a,err=_arma.ar_minnorm(x,
                          order=order, 
                          mode=mode,
                          lags=lags, 
                          signal_space = signal_space)
    
    psd=_arma.arma2psd(a,b=1,err=1,n_psd=n_psd)
    
    return psd

#-----------------------------------------
def kernel_minnorm(x, order, mode='full', 
                   kernel = 'linear', kpar = 1, 
                   lags=None, signal_space = False,
                   n_psd=None):
    '''  
  Estimation autoregression model based pseudo-spectrum 
        based on the minimum-norm (maximum-entropy) algorithm
        for kernel-matrix of the input signal.
    
    Parameters
    -----------
    * x:  1d ndarray. 
    * order: int.
        the autoregressive model (pole model) 
        order of the desired model. 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    * kernel: string,
        kernel type,
        kernel = {rbf,linear,poly,thin_plate,sigmoid,bump}.
    * kpar: kernel parameter,depends on the type.
    * lags: int or None,
        number of lags in the correlation matrix 
        (lags =x.shape[0]//2 if None).
    * signal_space: bool,
        if True, than coefficients will be estimated
        in the signal subspace, else in noise one.
    * n_psd: int or None,
        length of the desired pseudo-psectrum.
        
    Returns
    -------------
    * psd: 1d ndarray,
        estimated pseudo-spectrum.

    Notes
    -----------
    
    Example
    ------------
    
    References
    ------------
    [1a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page
    [2a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling
        
    ''' 
    x = _np.asarray(x)
    if n_psd is None: n_psd = x.shape[0]
    
    a,err=_arma.ar_kernel_minnorm(x,
                                 order = order, 
                                 mode  = mode,
                                 kernel = kernel, 
                                 kpar = kpar, 
                                 lags =lags, 
                                 signal_space = signal_space)
    
    psd=_arma.arma2psd(a,b=[1],err=err,n_psd=n_psd)
    
    return psd
#-------------------------------------
def hoyw(x, order, mode='full', lags=None, n_psd=None):
    '''
    Estimation of the pseudo-spectrum based on 
      the High order Yule-Walker (HOYW) 
      autoregression method.

    Parameters
    ------------
    * x:  1d ndarray. 
    * order: int.
        the autoregressive model (pole model) 
        order of the desired model. 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    * lags: int or None,
        number of lags in the correlation matrix 
        (lags =x.shape[0]//2 if None).
    * n_psd: int or None,
        length of the desired pseudo-psectrum.
        
    Returns
    -------------
    * psd: 1d ndarray,
        estimated pseudo-spectrum.

    References
    ------------------------
    [1] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [2] http://www2.ece.ohio-state.edu/~randy/SAtext/
        - Dr.Moses Spectral Analysis of Signals: Resource Page
        
    Examples
    ----------------

    '''    
    x = _np.asarray(x)
    if n_psd is None: n_psd = x.shape[0]    
    a,err=_arma.ar_hoyw(x,order=order, mode=mode, lags=lags)    
    psd  =_arma.arma2psd(a,b=1,err=1,n_psd=n_psd)    
    return psd
    
    