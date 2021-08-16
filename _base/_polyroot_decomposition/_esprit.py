import numpy as np
import scipy  

from ... import operators

__all__ = ['esprit','esprit_cor','kernel_esprit']
#------------------------------------
def esprit(x, order, mode='full',tls_rank = None):
    ''' 
    Signals decomposition based parameters estimation
      based on the Estimation of Signal Parameters via 
      Rotational Invariance Techniques (ESPRIT) algorithm.    
    
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

    Returns
    --------
    * components:  2d ndarray,
        signal components,
        dimnetion [x.size x order].
    
    Examples
    ----------
    
    Notes
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
    x = np.array(x)
    N=x.shape[0]

    #extract signal subspace
    R = operators.covariance_matrix(x, 
                                    lags=N//2, 
                                    mode=mode)

    D,V=np.linalg.eig(R)
    S=np.matrix(V[:,:order])

    S1=S[:-1,:]    
    S2=S[1:,:]

    if tls_rank is not None:
        S1, S2 = \
        operators.tls_turnication(S1,S2, 
                           tls_rank=max(tls_rank,order))

    Phi = scipy.linalg.lstsq(S1,S2)[0]

    roots,_=np.linalg.eig(Phi)

    return np.conj(roots)

#--------------------------------------------------
def esprit_cor(x, order, mode='full',cor_mode = 'full',tls_rank = None):
    ''' 
    Signals decomposition based parameters estimation
      based on the Estimation of Signal Parameters via 
      Rotational Invariance Techniques (ESPRIT) algorithm
      for additinally taken correlation function of input.    
    
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

    Returns
    --------
    * components:  2d ndarray,
        signal components,
        dimnetion [x.size x order].
    
    Examples
    ----------
    
    Notes
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
    x = np.array(x)
    N=x.shape[0]

    r = operators.correlation(x,mode=cor_mode)
    #extract signal subspace
    R = operators.covariance_matrix(r, 
                                    lags=N//2, 
                                    mode=mode)

    D,V=np.linalg.eig(R)
    S=np.matrix(V[:,:order])

    S1=S[:-1,:]    
    S2=S[1:,:]

    if tls_rank is not None:
        S1, S2 = \
        operators.tls_turnication(S1,S2, 
                           tls_rank=max(tls_rank,order))

    Phi = scipy.linalg.lstsq(S1,S2)[0]

    roots,_=np.linalg.eig(Phi)

    return np.conj(roots)
#------------------------------------------------------
def kernel_esprit(x, order, mode='full',kernel='linear',kpar=1,tls_rank = None):
    ''' 
    Signals decomposition based parameters estimation
      based on the Estimation of Signal Parameters via 
      Rotational Invariance Techniques (ESPRIT) algorithm
      taken for kernel matrix of input signal.    
    
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

    Returns
    --------
    * components:  2d ndarray,
        signal components,
        dimnetion [x.size x order].
    
    Examples
    ----------
    
    Notes
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
    x = np.array(x)
    N=x.shape[0]

    #extract signal subspace
    R = operators.kernel_matrix(x,
                                mode=mode,
                                kernel=kernel,
                                kpar=kpar,
                                lags=N//2,
                                ret_base=False,
                                normalization=True)

    D,V=np.linalg.eig(R)
    S=np.matrix(V[:,:order])

    S1=S[:-1,:]    
    S2=S[1:,:]

    if tls_rank is not None:
        S1, S2 = \
        operators.tls_turnication(S1,S2, 
                           tls_rank=max(tls_rank,order))

    Phi = scipy.linalg.lstsq(S1,S2)[0]

    roots,_=np.linalg.eig(Phi)

    return np.conj(roots)