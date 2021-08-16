
from ._base import _polyroot_decomposition as _polyroot
from ._base import _subspace 
from ._base import _arma 


__all__ = ['music',
           'ev',
           'phd',
           'phd_cor',
           'minvariance',
           'kernel_noisespace',
           'kernel_signalspace',
           'minnorm',
           'kernel_minnorm',
           'hoyw',
           'esprit',
           'esprit_cor',
           'kernel_esprit',
           'matrix_pencil',
           'matrix_pencil_cor',
           'matrix_pencil_cov',
           'kernel_matrix_pencil']

#----------------------------------------------
def music(x, order, mode='full', lags=None, fs=1):    
    '''  
    Estimation the pseudo-spectrum frequencies
      based on the MUltiple SIgnal Classification 
      (MUSIC) algorithm, determinated as noise 
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
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.

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
    noise_space = _subspace.music(x=x, order=order, mode=mode, lags=lags) 
    freqs = _subspace.subspace2freq(noise_space, order=order,fs=fs )
    return freqs

#-----------------------------------------
def ev(x, order, mode='full', lags=None,fs=1):    
    '''  
    Estimation the pseudo-spectrum frequencies 
      based on the Eigen Values (EV)algorithm, 
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
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.
    
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
    noise_space = _subspace.ev(x=x, order=order, mode=mode, lags=lags) 
    freqs = _subspace.subspace2freq(noise_space, order=order,fs=fs )
    return freqs

#------------------------------------
def phd(x, order, mode='full',fs=1):    
    '''  
    Estimation the pseudo-spectrum frequencies based on the 
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
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.
    
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
    noise_space = _subspace.pisarenko(x=x, order=order, mode=mode) 
    freqs = _subspace.subspace2freq(noise_space, order=order,fs=fs )
    return freqs

#------------------------------------
def phd_cor(x, order, mode='full', cor_mode = 'straight',fs=1):    
    '''  
    Estimation the pseudo-spectrum frequencies based on the 
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
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.
    
    Example
    -------------
        
        
    References
    -----------------
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
    noise_space = _subspace.pisarenko_cor(x=x, order=order, 
                                         mode=mode, cor_mode=cor_mode) 
    freqs = _subspace.subspace2freq(noise_space, order=order,fs=fs )
    return freqs

#------------------------------------
def minvariance(x, order, mode='full',fs=1):    
    '''  
    Estimation of the pseudo-spectrum frequencies based 
        on the minimum variance (maximum likelyhood) 
        algorithm, using normalized signal subspace. 

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
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.
    
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
    signal_space = _subspace.minvariance(x=x, order=order, mode=mode) 
    freqs = _subspace.subspace2freq(signal_space, order=order,fs=fs )
    return freqs

#------------------------------------
def kernel_noisespace(x, order, mode='full', 
                           kernel = 'linear', 
                           kpar=1, lags=None, 
                           use_ev = False,fs=1):    
    '''  
    Estimation the noise subspace based  
        pseudo-spectrum frequencies of the kernel matrix 
        taken for the input signal.

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
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.
    
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
    
    freqs = _subspace.subspace2freq(noise_space, order=order,fs=fs)
    return freqs

#------------------------------------
def kernel_signalspace(x, order, mode='full', 
                           kernel = 'linear', kpar=1, 
                           lags=None, use_ev = False, fs=1):    
    '''  
    Estimation the signal subspace based pseudo-spectrum  
        frequencies of the kernel based signal.

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
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.
    
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
    
    freqs = _subspace.subspace2freq(signal_space, order=order,fs=fs)
    return freqs
#---------------------------------------
def minnorm(x, order, mode='full',
             lags=None, signal_space = False, fs=1 ):
    '''  
    Estimation autoregression model based pseudo-spectrum 
        frequencies based on the minimum-norm 
        (maximum-entropy) algorithm.
    
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
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.
    
    Example
    ------------
    
    References
    ------------
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
    a,err=_arma.ar_minnorm(x,
                          order=order, 
                          mode=mode,
                          lags=lags, 
                          signal_space = signal_space)    
    freqs=_arma.ar2freq(a,order=order,fs=fs)    
    return freqs

#-----------------------------------------
def kernel_minnorm(x, order, mode='full', 
                        kernel = 'linear', kpar = 1, 
                        lags=None, signal_space = False,
                        fs=1):
    '''  
    Estimation autoregression model based pseudo-spectrum 
        frequencies based on the minimum-norm (maximum-entropy) 
        algorithm for kernel-matrix of the input signal.
    
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
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.
    
    Example
    ------------
    
    References
    ------------
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
    a,err=_arma.ar_kernel_minnorm(x,
                                 order = order, 
                                 mode  = mode,
                                 kernel = kernel, 
                                 kpar = kpar, 
                                 lags =lags, 
                                 signal_space = signal_space)   
    freqs=_arma.ar2freq(a,order=order,fs=fs)    
    return freqs
    
#-------------------------------------
def hoyw(x, order, mode='full', lags=None, fs=1):
    '''
    Estimation of the pseudo-spectrum frequencies
      based on the High order Yule-Walker (HOYW) 
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
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.

    References
    ------------------------
    [1a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/
        - Dr.Moses Spectral Analysis of Signals: 
        Resource Page.
        
    Examples
    ----------------

    '''    
    a,err=_arma.ar_hoyw(x,order=order, mode=mode, lags=lags)    
    freqs=_arma.ar2freq(a,order=order,fs=fs)    
    return freqs
#-----------------------------------------------    
def esprit(x, order, mode='full',tls_rank = None, fs=1):
    ''' 
    Frequency estimation based on the Estimation 
        of Signal Parameters via Rotational Invariance 
        Techniques (ESPRIT) algorithm.    
    
    Parameters
    ------------
    * x: input 1d ndarray.
    * order: int, 
        the model order.
    * mode: string, 
        mode for autoregression problem solving,
        mode = {full, traj,toeplitz, hankel, covar}.
    * tls_rank: int or None,
        if not None and >0, than Total Least Square
        (TLS) turnication with rank max(tls_rank,order) 
        will be perfermed.
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.
        
    
    Examples
    ----------

    References    
    -----------
    [1] Roy, Richard, and Thomas Kailath. 
        "ESPRIT-estimation of signal parameters 
        via rotational invariance techniques." 
        IEEE Transactions on acoustics, speech, 
        and signal processing 37, no. 7 
        (1989): 984-995.
    [2a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [2b] http://www2.ece.ohio-state.edu/~randy/SAtext/
        - Dr.Moses Spectral Analysis of Signals: 
        Resource Page. 
    '''
    roots = _polyroot.esprit(x, 
                            order=order, 
                            mode = mode, 
                            tls_rank=tls_rank)
    freqs = _polyroot.roots2freqs(roots=roots,
                                 order = order, fs = fs)
    return freqs

#--------------------------------------------------
def esprit_cor(x, order, mode='full',
               cor_mode = 'full',tls_rank = None, fs=1):
    ''' 
    Frequency estimation based on the Estimation 
        of Signal Parameters via Rotational Invariance 
        Techniques (ESPRIT) algorithm for additinally 
        taken correlation function of input.    
    
    Parameters
    ------------
    * x: input 1d ndarray.
    * order: int, 
        the model order.
    * mode: string, 
        mode for autoregression problem solving,
        mode = {full, traj,toeplitz, hankel, covar}.
    * cor_mode: string,
        additional correlation function,
        cor_mode = {same,full,straight}. 
    * tls_rank: int or None,
        if not None and >0, than Total Least Square
        (TLS) turnication with rank max(tls_rank,order) 
        will be perfermed.
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.
        
    
    Examples
    ----------

    References    
    -----------
    [1] Roy, Richard, and Thomas Kailath. 
        "ESPRIT-estimation of signal parameters 
        via rotational invariance techniques." 
        IEEE Transactions on acoustics, speech, 
        and signal processing 37, no. 7 
        (1989): 984-995.
    [2a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [2b] http://www2.ece.ohio-state.edu/~randy/SAtext/
        - Dr.Moses Spectral Analysis of Signals: 
        Resource Page. 
    '''
    roots = _polyroot.esprit_cor(x, 
                                order=order, 
                                mode = mode,
                                cor_mode = cor_mode,
                                tls_rank=tls_rank)
    freqs = _polyroot.roots2freqs(roots=roots,
                                 order = order, fs = fs)
    return freqs
#------------------------------------------------------
def kernel_esprit(x, order, mode='full',
                  kernel='linear',kpar=1,tls_rank = None, fs=1):
    ''' 
    Frequency estimation based on the Estimation 
        of Signal Parameters via Rotational Invariance 
        Techniques (ESPRIT) algorithm taken for an 
        kernel matrix of input signal.    
    
    Parameters
    ------------
    * x: input 1d ndarray.
    * order: int, 
        the model order.
    * mode: string, 
        mode for autoregression problem solving,
        mode = {full, traj,toeplitz, hankel, covar}.
    * kernel: string,        
        kernel type 
        kernel = {rbf, linear, poly, thin_plate, sigmoid}.
    * kpar: float,
        kernel parameter, depends on the kernel type.
    * tls_rank: int or None,
        if not None and >0, than Total Least Square
        (TLS) turnication with rank max(tls_rank,order) 
        will be perfermed.
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.
        
    
    Examples
    ----------

    References    
    -----------
    [1] Roy, Richard, and Thomas Kailath. 
        "ESPRIT-estimation of signal parameters 
        via rotational invariance techniques." 
        IEEE Transactions on acoustics, speech, 
        and signal processing 37, no. 7 
        (1989): 984-995.
    [2a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [2b] http://www2.ece.ohio-state.edu/~randy/SAtext/
        - Dr.Moses Spectral Analysis of Signals: 
        Resource Page. 
    
    '''
    roots = _polyroot.kernel_esprit(x, 
                                   order=order,
                                   mode = mode,
                                   kernel=kernel, 
                                   kpar=kpar,
                                   tls_rank=tls_rank)
    freqs = _polyroot.roots2freqs(roots=roots,
                                 order = order, fs = fs)
    return freqs

#-------------------------------------------------------------
def matrix_pencil(x, order, mode='full', tls_rank = None, fs =1):
    ''' 
    Matrix Pencil Method (MPM) 
        for of the decay signals model
        frequencies estimation.
    
    Parameters
    --------------
    * x: input 1d ndarray.    
    * order: int,
        the model order.    
    * mode:  string,
        mode = {full, toeplitz, hankel, covar, traj},
        mode for autoregression problem solving.   
    * tls_rank: int or None,
        rank of Total Least-Square turnication of
        the processied matrix, if not None,
        tls_rank = min(max(order,tls_rank),N).
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.
    * MPM is calculated for each component as follows:
     ..math::    
     s_p(n)=[v_N^T * λ_p]^(-1)*x(n)*λ_p^n,
     where 
     * s_p(n) is the p-th component of decomposition;
     * v^N is the Vandermonde matrix operator 
                         of degrees from 0 to N-1; 
     * λ_p is the p-th eigenvalue of matrix r_1^(-1)(n)*r_2(n),  
        where r_1(n) and r_2(n) are rows from 0 to P-1 (r_1(n)) 
        and rows from 1 to P (r_2(n)) 
       of the transported lags_matrix of x(n) with order P+1.  

    Examples
    -----------
    
    
    References
    -----------
    [1a] A. Fernandez-Rodriguez, L. de Santiago, 
    M.E.L. Guillen, et al.,"Coding Prony’s method in 
    MATLAB and applying it to biomedical signal filtering", 
        BMC Bioinformatics, 19, 451 (2018).
    [1b]  https://bmcbioinformatics.biomedcentral.com/
    articles/10.1186/s12859-018-2473-y 
     
    '''
    roots = _polyroot.matrix_pencil(x, 
                                   order = order, 
                                   mode  = mode,
                                   tls_rank=tls_rank)
    freqs = _polyroot.roots2freqs(roots=roots,
                                 order = order, fs = fs)
    return freqs

#--------------------------------------
def matrix_pencil_cor(x, order, mode='full',
                      cor_mode='same', tls_rank = None, fs=1):
    ''' 
    Matrix Pencil Method (MPM) 
        for of the decay signals model
        frequencies estimation for 
        additionally taken correlation function.
    
    Parameters
    --------------
    * x: input 1d ndarray.    
    * order: int,
        the model order.    
    * mode:  string,
        mode = {full, toeplitz, hankel, covar, traj},
        mode for autoregression problem solving.   
    * cor_mode: string,
        additional correlation function,
        cor_mode = {same,full,straight}.
    * tls_rank: int or None,
        rank of Total Least-Square turnication of
        the processied matrix, if not None,
        tls_rank = min(max(order,tls_rank),N).
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.
    * MPM is calculated for each component as follows:
     ..math::    
     s_p(n)=[v_N^T * λ_p]^(-1)*x(n)*λ_p^n,
     where 
     * s_p(n) is the p-th component of decomposition;
     * v^N is the Vandermonde matrix operator 
                         of degrees from 0 to N-1; 
     * λ_p is the p-th eigenvalue of matrix r_1^(-1)(n)*r_2(n),  
        where r_1(n) and r_2(n) are rows from 0 to P-1 (r_1(n)) 
        and rows from 1 to P (r_2(n)) 
       of the transported lags_matrix of x(n) with order P+1.  

    Examples
    -----------
    
    
    References
    -----------
    [1a] A. Fernandez-Rodriguez, L. de Santiago, 
    M.E.L. Guillen, et al.,"Coding Prony’s method in 
    MATLAB and applying it to biomedical signal filtering", 
        BMC Bioinformatics, 19, 451 (2018).
    [1b]  https://bmcbioinformatics.biomedcentral.com/
    articles/10.1186/s12859-018-2473-y 
     
    '''
    roots = _polyroot.matrix_pencil_cor(x, 
                                       order = order, 
                                       mode  = mode,
                                       cor_mode = cor_mode,
                                       tls_rank=tls_rank)
    freqs = _polyroot.roots2freqs(roots=roots,
                                 order = order, fs = fs)
    return freqs

#-------------------------------------------------------------
def matrix_pencil_cov(x, order, mode, tls_rank = None, fs=1):
    ''' 
    Matrix Pencil Method (MPM) 
        for of the decay signals model
        frequencies estimation for 
        the signal covariance matrix.
    
    Parameters
    --------------
    * x: 1d ndarray.    
    * order: int,
        the model order.    
    * mode:  string,
        mode = {full, toeplitz, hankel, covar, traj},
        mode for autoregression problem solving.   
    * tls_rank: int or None,
        rank of Total Least-Square turnication of
        the processied matrix, if not None,
        tls_rank = min(max(order,tls_rank),N).
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.
    * MPM is calculated for each component as follows:
     ..math::    
     s_p(n)=[v_N^T * λ_p]^(-1)*x(n)*λ_p^n,
     where 
     * s_p(n) is the p-th component of decomposition;
     * v^N is the Vandermonde matrix operator of degrees from 0 to N-1; 
     * λ_p is the p-th eigenvalue of matrix r_1^(-1)(n)*r_2(n),  
        where r_1(n) and r_2(n) are rows from 0 to P-1 (r_1(n)) 
        and rows from 1 to P (r_2(n)) 
       of the transported lags_matrix of x(n) with order P+1.  

    Examples
    -----------
    
    
    References
    -----------
    [1a] A. Fernandez-Rodriguez, L. de Santiago, 
    M.E.L. Guillen, et al.,"Coding Prony’s method in 
    MATLAB and applying it to biomedical signal filtering", 
        BMC Bioinformatics, 19, 451 (2018).
    [1b]  https://bmcbioinformatics.biomedcentral.com/
    articles/10.1186/s12859-018-2473-y 
     
    '''
    roots = _polyroot.matrix_pencil_cov(x, 
                                       order = order, 
                                       mode  = mode,
                                       tls_rank=tls_rank)
    freqs = _polyroot.roots2freqs(roots=roots,
                                 order = order, fs = fs)
    return freqs
#-----------------------------------------
def kernel_matrix_pencil(x, order, mode, kernel = 'rbf', 
                         kpar=1, tls_rank = None, fs = 1):
    ''' 
    Matrix Pencil Method (MPM) 
        for of the decay signals model
        frequencies estimation for 
        the signal kernel matrix.
    
    Parameters
    --------------
    * x: input 1d ndarray.    
    * order: int,
        the model order.    
    * mode:  string,
        mode = {full, toeplitz, hankel, covar, traj},
        mode for autoregression problem solving.   
    * kernel: string,
        kernel = {linear,rbf,thin_plate,bump,poly,sigmoid},
        kernel mode.
    * kpar: float,
        kernel parameter depends on the kernel type.
    * tls_rank: int or None,
        rank of Total Least-Square turnication of
        the processied matrix, if not None,
        tls_rank = min(max(order,tls_rank),N).
    * fs: int,
        sampling frequency.
    
    Returns
    -------------
    * freqs: 1d ndarray,
        estimated pseudo-spectrum frequencies.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * For obtaining frequency values in points
        use fs = x.size.
    * MPM is calculated for each component as follows:
     ..math::    
     s_p(n)=[v_N^T * λ_p]^(-1)*x(n)*λ_p^n,
     where 
     * s_p(n) is the p-th component of decomposition;
     * v^N is the Vandermonde matrix operator of degrees from 0 to N-1; 
     * λ_p is the p-th eigenvalue of matrix r_1^(-1)(n)*r_2(n),  
        where r_1(n) and r_2(n) are rows from 0 to P-1 (r_1(n)) 
        and rows from 1 to P (r_2(n)) 
       of the transported lags_matrix of x(n) with order P+1.  

    Examples
    -----------
    
    
    References
    -----------
    [1a] A. Fernandez-Rodriguez, L. de Santiago, 
    M.E.L. Guillen, et al.,"Coding Prony’s method in 
    MATLAB and applying it to biomedical signal filtering", 
        BMC Bioinformatics, 19, 451 (2018).
    [1b]  https://bmcbioinformatics.biomedcentral.com/
    articles/10.1186/s12859-018-2473-y 
     
    '''
    roots = _polyroot.kernel_matrix_pencil(x, 
                                          order = order, 
                                          mode = mode,
                                          kernel=kernel, 
                                          kpar=kpar, 
                                          tls_rank=tls_rank)
    freqs = _polyroot.roots2freqs(roots=roots,
                                 order = order, fs = fs)
    return freqs