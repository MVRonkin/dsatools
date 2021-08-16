import numpy as np
import scipy  

from ... import operators

__all__ = ['dmd','dmd_fb','kernel_dmd']
#------------------------------------------------------------------------
def dmd(x, order= None, mode='traj', 
        exact = True, tls_rank =  None, lags = None):
    '''
    Dynamic mode decomposition (DMD).
    
    Parameters
    -------------
    * x:  1d ndarray,
        input signal.
    * order: int,
        order of the model (number of valuable components, 
        size of signal subspace). 
    * mode:  string,        
        The mode of correlation function (traj as default),
        mode = {traj, full, covar, toeplitz, hankel}.
    * exact: bool,
        Exect problem solution if true, standard otherwise.
    * tls_rank: int or None,
        if not None and >0, than TLS processing 
        will be carried out in additional (TLS_DMD). 
    * lags: int or None,
        number of lags in the out put samples, 
                                if None, then x.size//2.
                                
    Returns
    --------
    * components:  2d ndarray,
        dimnetion [x.size x order], signal components.
    
    Notes
    ------------
    * See for svd rank estimation methods for case order is None.
    
    Referenes
    ------------
    [1] J.H. Tu, et al. "On Dynamic Mode Decomposition: 
        Theory and Applications" (2013).
        (available at `arXiv <http://arxiv.org/abs/1312.0041>`_).  
    [2a] N.B. Erichson and C. Donovan.
        "Randomized Low-Rank Dynamic Mode Decomposition 
        for Motion Detection" (2015).
    [2b] https://github.com/erichson/DMDpack
    [3]  https://github.com/mathLab/PyDMD
    
    '''
    x = np.asarray(x)
    N = x.shape[0]
    if lags is None: lags = N//2
        
    snap = operators.covariance_matrix(x,
                                       mode=mode,
                                       lags = lags) #snapshot

    X = snap[ :, :-1] 
    Y = snap[ :, 1: ] 
    
    X, Y = operators.tls_turnication(X,Y,tls_rank)# if tls rank>0 and not None

    U, s, Vh = scipy.linalg.svd( X )

    if order is None: 
        order = s.size
        
    U = U[:,:order]
    s = s[:order]
    Vh = Vh[:order,:]
    
    G = np.dot( Y , np.conj(Vh.T)/s ) 
    M = np.dot( np.conj(U.T), G ) # low rank operator

    _, W = scipy.linalg.eig( M )    
 
    if exact: 
        modes = np.dot( G , W )
    
    else: # standard solution 
        modes = np.dot( U , W )    

    return np.conj(modes.T)

#------------------------------------------------------------------------
def dmd_fb(x, order= None, mode='traj', 
           exact = True, tls_rank =  None, lags = None):
    '''
    Force-backward Dynamic mode decomposition (DMD).
    
    Parameters
    -------------
     * x:  1d ndarray,
        input signal.
    * order: int,
        order of the model (number of valuable components, 
        size of signal subspace). 
    * mode:  string,        
        The mode of correlation function (traj as default),
        mode = {traj, full, covar, toeplitz, hankel}.
    * exact: bool,
        Exect problem solution if true, standard otherwise.
    * tls_rank: int or None,
        if not None and >0, than TLS processing 
        will be carried out in additional (TLS_DMD). 
    * lags: int or None,
        number of lags in the out put samples, 
                                if None, then x.size//2.
                                
    Returns
    --------
    * components:  ndarray,
        dimnetion [x.size x order], signal components.
    
    Notes
    ----------
    * See for svd rank estimation methods for case order is None.
    
    Referenes
    -----------
    [1] J.H. Tu, et al. "On Dynamic Mode Decomposition: 
        Theory and Applications" (2013).
    (available at `arXiv <http://arxiv.org/abs/1312.0041>`_).  
    [2a] N.B. Erichson and C. Donovan.
        "Randomized Low-Rank Dynamic Mode Decomposition 
        for Motion Detection" (2015).
    [3]  https://github.com/mathLab/PyDMD
    
    '''    
    x = np.asarray(x)
    N = x.shape[0]
    if lags is None: lags = N//2
        
    snap = operators.covariance_matrix(x,
                                       mode=mode, 
                                       lags = lags) #snapshot

    X = snap[ :, :-1] 
    Y = snap[ :, 1: ] 
    
    X,Y = operators.tls_turnication(X,Y,tls_rank)# if tls rank>0 and not None

    Ux, sx, Vhx = scipy.linalg.svd( X )
    Uy, sy, Vhy = scipy.linalg.svd( Y ) 

    if order is None: 
        order = s.size
    
    #force branch   
    Ux = Ux[:,:order]
    sx = sx[:order]
    Vhx = Vhx[:order,:]

    Gf = np.dot( Y , np.conj(Vhx.T) /sx ) 
    Mf = np.dot( np.conj(Ux.T), Gf ) # fAtilda
    
    #back branch
    Uy = Uy[:,:order]
    sy = sy[:order]
    Vhy = Vhy[:order,:]    
  
    Gb = np.dot( X , np.conj(Vhy.T) /sy ) 
    Mb = np.dot( np.conj(Uy.T), Gb ) #  bAtilda    
    
    # low rank operator, Atilda  
    M = scipy.linalg.sqrtm(Mf.dot(np.linalg.inv(Mb)))
    _, W = scipy.linalg.eig( M )    
 
    if exact: 
        modes = np.dot( Gf , W )  
    
    else: # exact 
        modes = np.dot( Ux , W )  

    return np.conj(modes.T)

#------------------------------------------------------------------------
def kernel_dmd(x, order = None,  mode = 'full', 
               kernel   = 'rbf', kpar = 1, 
               exact    = True,  fb   = False, 
               tls_rank = None,  lags = None):
    '''
    Kernel Dynamic mode decomposition (DMD).

    Parameters
    -------------
     * x:  1d ndarray,
        input signal.
    * order: int,
        order of the model (number of valuable components, 
        size of signal subspace). 
    * mode:  string,        
        The mode of correlation function (traj as default),
        mode = {traj, full, covar, toeplitz, hankel}.
    * kernel: string,
        kernel type: {linear, rbf, poly, thin_plate, bump}.
    * kpar: float,
        kernel parameter.
    * exact: bool,
        Exect problem solution if true, standard otherwise.
    * fb: bool,
        if True, than force-back dmd will be taken.
    * tls_rank: int or None,
        if not None and >0, than TLS processing 
        will be carried out in additional (TLS_DMD). 
    * lags: int or None,
        number of lags in the out put samples, 
                                if None, then x.size//2.
                                
    Returns
    --------
    * components:  ndarray,
        signal components matrix 
        with dimnetion [x.size x order].

    
    Notes
    ----------
    * See for svd rank estimation methods for case order is None.
    
    Referenes
    ----------
    [1] J.H. Tu, et al. "On Dynamic Mode Decomposition: 
        Theory and Applications" (2013).
        (available at `arXiv <http://arxiv.org/abs/1312.0041>`_).
    '''    
    
    x = np.asarray(x)
    N = x.shape[0]
    if lags is None: lags = N//2
    
    snap  = operators.kernel_matrix(x,
                                    mode  =mode,
                                    kernel=kernel,
                                    kpar  =kpar,
                                    lags  = lags) #snapshot
    
    #TODO: check if remove normalization.
    #predicted part without kernal
    snap1 = operators.kernel_matrix(x,
                                    mode  =mode,
                                    kernel='linear',
                                    lags  = lags)  
    X = snap[ :, :-1] 
    Y = snap1[ :, 1: ] 
    
    X, Y = operators.tls_turnication(X,Y,tls_rank)# if tls rank>0 and not None

    U, s, Vh = scipy.linalg.svd( X )

    if order is None: 
        order = s.size
        
    U = U[:,:order]
    s = s[:order]
    Vh = Vh[:order,:]
    
    G = np.dot( Y , np.conj(Vh.T)/s ) 
    M = np.dot( np.conj(U.T), G ) # low rank operator
    
    if fb:
        #back branch
        Uy, sy, Vhy = scipy.linalg.svd( Y )
        Uy = Uy[:,:order]
        sy = sy[:order]
        Vhy = Vhy[:order,:]    
        Mb = np.linalg.multi_dot([np.conj(Uy.T),X,np.conj(Vhy.T)/sy]) #  bAtilda    
        # low rank operator, Atilda  
        M = scipy.linalg.sqrtm(M.dot(np.linalg.inv(Mb)))
   
    _, W = scipy.linalg.eig( M )    
 
    if exact: 
        modes = np.dot( G , W )
    
    else: # standard solution 
        modes = np.dot( U , W )    

    return np.conj(modes.T)

# #------------------------------------------------------------------------
# def _tls_turnication(X, Y, tlsq_rank=0):
#     ''' 
#     Total Least Square turnication.
    
#     References:
#     https://github.com/mathLab/PyDMD/blob/master/pydmd    
#     '''
#     if tlsq_rank in [0,None]: return X, Y
    
#     _,_,Vh = np.linalg.svd(np.append(X, Y, axis=0), full_matrices=False)
    
#     Vh  = Vh[: np.min([tlsq_rank, Vh.shape[0]]), :]

#     VV    = np.dot(np.conj(Vh.T),Vh)
    
#     return X.dot(VV), Y.dot(VV)   