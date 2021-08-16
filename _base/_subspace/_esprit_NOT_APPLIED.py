import numpy as np
import scipy  

__all__ = ['esprit','kernel_esprit']
from ... import matrix


################
''' SEARCH IT IN THE TIME_DOMAIN SECTION ''' 
#################
#----------------------------------------------------------------
def esprit(x, order, mode='full',fs=1, tls_rank = None):
    ''' 
    SEARCH IT IN THE TIME_DOMAIN SECTION
    
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
    * fs: float,
        sampling frequency.
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
    '''
    x = np.array(x)
    N=x.shape[0]
        
    if fs is None:fs = N

    #extract signal subspace
    R = matrix.covariance_matrix(x, lags=N//2, 
                                 mode=mode, fb=False)

    D,V=np.linalg.eig(R)
    S=np.matrix(V[:,:order])

    #Remove last row
    S1=S[:-1,:]
    #Remove first row
    S2=S[1:,:]

    if tls_rank is not None:
        S1, S2 = \
        matrix.tls_turnication(S1,S2, tls_rank=max(tls_rank,order))

    Phi = np.dot(np.linalg.pinv(S1),S2)
#     Phi=(S1.H*S1).I*S1.H*S2

    #Perform eigenvalue decomposition
    roots,_=np.linalg.eig(Phi)

    angle=np.angle(roots)
    
    #frequency normalisation
    f=-fs*angle/(2.*np.pi)
    
    return f

#-------------------------------------------------------------------
def kernel_esprit(x, order, mode='full',kernel='linear',
                          kpar=1,fs=None, tls_rank = None):
    ''' 
    
    SEARCH IT IN THE TIME_DOMAIN SECTION
     
    PSD frequency estimation by Estimate the frequency components 
    based on the kernel Estimation of Signal Parameters via 
    Rotational Invariance Techniques (ESPRIT) algorithm, 
    optionally using Total Least Square (TLS) solution.  

    Parametres
    ------------
    * x: 1d ndarray of size N
    * order: number of extracted components.
    * mode:  mode for data matrix.   
    * kernel: kernel types:
        {rbf,poly,sigmoid,thin_plate,linear}.
    * kpar: kernel parameter.
    * fs: sampling frequency (x.size as defaults).
    * tls_rank: if not None, than tls turnication with rank
        max(tls_rank,order) will be perfermed. 
    
    Returns
    ------------
    * 1d ndarray containing the estimated frequencies.
    
    Notes
    ----------------
    
    '''
    x = np.array(x)
    N=x.shape[0]
        
    if fs is None:fs = N

    #extract signal subspace
    R = matrix.kernel_matrix(x, 
                             lags=N//2, 
                             ktype=kernel,
                             kpar = kpar,
                             mode=mode)

    D,V=np.linalg.eig(R)
    S=np.matrix(V[:,:order])

    #Remove last row
    S1=S[:-1,:]
    #Remove first row
    S2=S[1:,:]

    if tls_rank is not None:
        S1, S2 = \
        matrix.tls_turnication(S1,S2, 
                               tls_rank=max(tls_rank,order))

    Phi = np.dot(np.linalg.pinv(S1),S2)
#     Phi=(S1.H*S1).I*S1.H*S2

    #Perform eigenvalue decomposition
    D,U=np.linalg.eig(Phi)

    angle=np.angle(D)
    
    #frequency normalisation
    f=-fs*angle/(2.*np.pi)
    
    return f