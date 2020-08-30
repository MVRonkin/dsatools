from ._base import _polyroot_decomposition as _polyroot

from ._base._imf_decomposition import(emd, 
                                      ewt, 
                                      vmd, 
                                      hvd)
from ._base._time_domain_decomposition import(pca,
                                              pca_cor, 
                                              kernel_pca,
                                              dmd, 
                                              dmd_fb, 
                                              kernel_dmd,
                                              ssa,
                                              kernel_ssa)

__all__ = ['emd', 
           'ewt', 
           'vmd', 
           'hvd',
           'pca',
           'pca_cor', 
           'kernel_pca',
           'dmd', 
           'dmd_fb', 
           'kernel_dmd',
           'ssa',
           'kernel_ssa',
           'esprit',
           'esprit_cor',
           'kernel_esprit',
           'matrix_pencil',
           'matrix_pencil_cor',
           'matrix_pencil_cov',
           'kernel_matrix_pencil']


#-----------------------------------------------    
def esprit(x, order, mode='full',tls_rank = None):
    ''' 
    Signal decomposition based on the Estimation 
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
    
    Returns
    -------------
    * components, 2d ndarray 
        components with dimentions 
        (order, x.shape[0]).
   
    Notes
    -----------
        
    
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
    psds = _polyroot.roots2decomposition(x,roots=roots,order = order)
    return psds

#--------------------------------------------------
def esprit_cor(x, order, mode='full',
               cor_mode = 'full',tls_rank = None):
    ''' 
    Signal decomposition based on the Estimation 
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
    
    Returns
    -------------
    * components, 2d ndarray 
        components with dimentions 
        (order, x.shape[0]).
   
    Notes
    -----------
        
    
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
    psds = _polyroot.roots2decomposition(x,roots=roots,order = order)
    return psds
#------------------------------------------------------
def kernel_esprit(x, order, mode='full',
                  kernel='linear',kpar=1,tls_rank = None):
    ''' 
    Signal decomposition based on the Estimation 
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
    
    Returns
    -------------
    * components, 2d ndarray 
        components with dimentions 
        (order, x.shape[0]).
   
    Notes
    -----------
        
    
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
    psds = _polyroot.roots2decomposition(x,roots=roots,order = order)
    return psds

#-------------------------------------------------------------
def matrix_pencil(x, order, mode='full', tls_rank = None):
    ''' 
    Signal decomposition based on the 
        Matrix Pencil Method (MPM) for of the 
        decay signals model.
    
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
    
    Returns
    -------------
    * components, 2d ndarray 
        components with dimentions 
        (order, x.shape[0]).
   
    Notes
    -----------
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
    psds = _polyroot.roots2decomposition(x,roots=roots,order = order)
    return psds

#--------------------------------------
def matrix_pencil_cor(x, order, mode='full',
                      cor_mode='same', tls_rank = None):
    ''' 
    Signal decomposition based on the 
        Matrix Pencil Method (MPM) for the 
        correlation function of the decay signals model.
    
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
    
    Returns
    -------------
    * components, 2d ndarray 
        components with dimentions 
        (order, x.shape[0]).
   
    Notes
    -----------
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
    psds = _polyroot.roots2decomposition(x,roots=roots,order = order)
    return psds

#-------------------------------------------------------------
def matrix_pencil_cov(x, order, mode, tls_rank = None):
    ''' 
    Signal decomposition based on the 
        Matrix Pencil Method (MPM) for the 
        covariance matrix of the decay signals model.
    
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
    
    Returns
    -------------
    * components, 2d ndarray 
        components with dimentions 
        (order, x.shape[0]).
   
    Notes
    -----------
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
    roots = _polyroot.matrix_pencil_cov(x, order = order, 
                                      mode=mode, tls_rank=tls_rank)
    psds = _polyroot.roots2decomposition(x,roots=roots,order = order)
    return psds
#-----------------------------------------
def kernel_matrix_pencil(x, order, mode, kernel = 'rbf', 
                                 kpar=1, tls_rank = None):
    ''' 
    Signal decomposition based on the 
        Matrix Pencil Method (MPM) for the 
        kernel matrix of the decay signals model.
    
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
    
    Returns
    -------------
    * components, 2d ndarray 
        components with dimentions 
        (order, x.shape[0]).
   
    Notes
    -----------
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
    roots = _polyroot.kernel_matrix_pencil(x, order = order, mode = mode,
                                          kernel=kernel, kpar=kpar, tls_rank=tls_rank)
    psds = _polyroot.roots2decomposition(x,roots=roots,order = order)
    return psds