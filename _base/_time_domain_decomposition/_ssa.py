import numpy as np
import scipy  

from ... import operators

__all__ = ['ssa','kernel_ssa']
#-----------------------------------------------------------------
__EPSILON__ = 1e-4
def ssa(x, order, mode='toeplitz', lags=None, averaging=True, extrasize = False):    
    '''  
    Estimation the signal components based on the 
        Singular Spectral Analysis (SSA) algorithm.
       
    Parameters
    -------------
    * x: 1d ndarray,    
        input 1d signal.
    * order: int,
        order of the model (number of valuable components, 
        size of signal subspace). 
    * mode:  
        The mode of lags matrix 
        (i.e. trajectory (or caterpillar) matrix or its analouge),
        mode = {traj, full, covar, toeplitz, hankel}.
    * lags: int or None,
        Number of lags in correlation function 
        (x.shape[0]//2 by default). 
    * averaging: bool,
        If True, then mean of each diagonal will be taken 
        for diagonal averaging instead of 
        just summarizing (True, by default).
    * extrasize: bool,
        if True, than near doubled size of 
            output will be returned.
    
    Returns
    -----------
    * components, 2d ndarray 
        components with dimentions 
        (order, x.shape[0]).
      
    Notes
    ------------
    * Not all methods combinations are tested.    
    * SSA is calculated for each component as:
      ..math::
      s_p(n)=dav{λ_p^0.5*U_p(n)*[(r(n)^T*U_p(n))⁄λ_p^0.5]^T},
      where 
      * s_p(n) is the p-th component of ssa decomposition;
      * U_p(n) and λ_p are eigenvectors and eigenvalues 
         correspondingly for matrix rr^H;
          where 
          * r(n) is lags_matrix formed for x(t).
      * dav is the operator of averaging by each diagonal. 

    
    Refernces
    ------------
    [1] A. Zhigljavsky, Singular Spectrum Analysis 
            for Time Series. In: Lovric M. (eds) 
            International Encyclopedia of Statistical 
            Science. Springer, Berlin, Heidelberg, 2011.
    
    See also
    -------------
    kernel_ssa, 
    pca, 
    dmd, 
    matrix_pencil,
    esprit

    '''   
    x = np.asarray(x)
    N = x.shape[0]       
    
    #TODO: for toeplitz and hankel Nlags always = N
    if(lags is None): lags = N//2
    
    reverse = False
    if(mode in ['traj', 'hankel', 'trajectory','caterpillar']):
        reverse = True
        
    base = operators.lags_matrix(x, 
                                 lags=lags, 
                                 mode=mode)
    R = np.dot(base.T,np.conj(base))

    # TODO: in my practice eigen value always sorted
    # from the highest to the lowest, 
    # but probably sorting would be better
    es,ev=np.linalg.eig(R)   

#     if(use_eigval): 
    es = np.sqrt(es)+__EPSILON__
#     else: 
#     es = np.ones(es.shape)


    psd = np.zeros((order, 
                    base.shape[0] + base.shape[1] -1 ),
                    dtype = x.dtype)
        
    for i in range(order):        
        Ys = np.matrix(ev[:,i])*es[i]
        Vs = np.dot(base, Ys.H)/es[i]       
        hankel = np.outer(Ys,Vs)
        
        diag= operators.diaganal_average(hankel,
                                         reverse=reverse,
                                         averaging=averaging,
                                         samesize=extrasize)
        
        psd[i,:diag.size] = diag
    
    #TODO: calc diag.size it vale in psd declaration
    psd = psd[:,:diag.size]
    
    
    if(mode in ['traj', 'trajectory','caterpillar']):
        psd = np.conj(psd)

    return  np.asarray(psd)/N
#------------------------------------------
__EPSILON__ = 1e-4
def kernel_ssa(x, order, mode='toeplitz', kernel='linear',kpar=1,
                    lags=None, averaging=True, extrasize = False):    
    '''  
    Estimation the signal components based on the 
        Singular Spectral Analysis (SSA) algorithm.
       
    Parameters
    -------------
    * x: 1d ndarray,    
        input 1d signal.
    * order: int,
        order of the model (number of valuable components, 
        size of signal subspace). 
    * mode:  
        The mode of lags matrix 
        (i.e. trajectory (or caterpillar) matrix or its analouge),
        mode = {traj, full, covar, toeplitz, hankel}.
    * kernel: string,
        kernel matrix type,
        kernel = {'rbf','thin_plate','linear','euclid',
                 'minkowsky','sigmoid','poly'}.      
    * kpar: float,
        is kernal parameter, depends on kernal type.
    * lags: int or None,
        Number of lags in correlation function 
        (x.shape[0]//2 by default). 
    * averaging: bool,
        If True, then mean of each diagonal will be taken 
        for diagonal averaging instead of 
        just summarizing (True, by default).
    * extrasize: bool,
        if True, than near doubled size of 
            output will be returned.
    
    Returns
    -----------
    * components, 2d ndarray 
        components with dimentions 
        (order, x.shape[0]).
      
    Notes
    ------------
    * Not all methods combinations are tested.    
    * SSA is calculated for each component as:
      ..math::
      s_p(n)=dav{λ_p^0.5*U_p(n)*[(r(n)^T*U_p(n))⁄λ_p^0.5]^T},
      where 
      * s_p(n) is the p-th component of ssa decomposition;
      * U_p(n) and λ_p are eigenvectors and eigenvalues 
         correspondingly for matrix rr^H;
          where 
          * r(n) is lags_matrix formed for x(t).
      * dav is the operator of averaging by each diagonal. 

    
    Refernces
    ------------
    [1] A. Zhigljavsky, Singular Spectrum Analysis 
            for Time Series. In: Lovric M. (eds) 
            International Encyclopedia of Statistical 
            Science. Springer, Berlin, Heidelberg, 2011.
    
    See also
    -------------
    ssa, 
    pca, 
    dmd, 
    matrix_pencil,
    esprit

    '''   
    x = np.asarray(x)
    N = x.shape[0]       
    
    #TODO: for toeplitz and hankel Nlags always = N
    if(lags is None): lags = N//2
    
    reverse = False
    if(mode in ['traj', 'hankel', 'trajectory','caterpillar']):
        reverse = True
        
    R,base = operators.kernel_matrix(x,
                                     mode   = mode,
                                     kernel = kernel,
                                     kpar   = kpar,
                                     lags   = lags,
                                     ret_base=True)

    # TODO: in my practice eigen value always sorted
    # from the highest to the lowest, 
    # but probably sorting would be better
    es,ev=np.linalg.eig(R)   

#     if(use_eigval): 
    es = np.sqrt(es)+__EPSILON__
#     else: 
#     es = np.ones(es.shape)


    psd = np.zeros((order, 
                    base.shape[0] + base.shape[1] -1 ),
                    dtype = x.dtype)
        
    for i in range(order):        
        Ys = np.matrix(ev[:,i])*es[i]
        Vs = np.dot(base, Ys.H)/es[i]       
        hankel = np.outer(Ys,Vs)
        
        diag= operators.diaganal_average(hankel,
                                         reverse=reverse,
                                         averaging=averaging,
                                         samesize=extrasize)
        
        psd[i,:diag.size] = diag
    
    #TODO: calc diag.size it vale in psd declaration
    psd = psd[:,:diag.size]
    
    
    if(mode in ['traj', 'trajectory','caterpillar']):
        psd = np.conj(psd)

    return  np.asarray(psd)/N
