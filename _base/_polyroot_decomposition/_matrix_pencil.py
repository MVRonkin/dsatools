import numpy as np
import scipy  

from ... import operators


__all__ = ['matrix_pencil','matrix_pencil_cor','matrix_pencil_cov', 'kernel_matrix_pencil']
#-------------------------------------------------------------
def matrix_pencil(x, order, mode='full', tls_rank = None):
    ''' 
    Matrix Pencil Method (MPM) 
        for of the decay signals model
        parameters estimation.
    
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
    ------------
    * roots: 1d ndarray,
        signal parameters in roots form.
    
    Notes
    ----------------
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
    x = np.asarray(x)
    N = x.shape[0]

    mtx = operators.lags_matrix(x,
                                mode=mode,
                                lags = order+1)
    mtx1 = mtx[:-1,:]
    mtx2 = mtx[1:,:]
    
    if tls_rank:
        _tls_rank = min(max(order,tls_rank),N)
        mtx1, mtx2 \
            = operators.tls_turnication(mtx1, mtx2, _tls_rank)

    QZ=scipy.linalg.lstsq(mtx1,mtx2)[0]
    
    roots, _ =  np.linalg.eig(QZ)    

    if mode in ['toeplitz']:roots = np.conj(roots)
    return roots

#--------------------------------------
def matrix_pencil_cor(x, order, mode='full',cor_mode='same', tls_rank = None):
    ''' 
    Matrix Pencil Method (MPM) 
        for of the decay signals model
        parameters estimation for 
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

    Returns
    ------------
    * roots: 1d ndarray,
        signal parameters in roots form.
    
    Notes
    ----------------
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
    x = np.asarray(x)
    N = x.shape[0]

    r = operators.correlation(x,
                              mode=cor_mode) 
    
    mtx = operators.lags_matrix(r,
                                mode=mode,
                                lags = order+1)
    mtx1 = mtx[:-1,:]
    mtx2 = mtx[1:,:]
    
    if tls_rank:
        _tls_rank = min(max(order,tls_rank),N)
        mtx1, mtx2 \
            = operators.tls_turnication(mtx1, mtx2, _tls_rank)

    QZ=scipy.linalg.lstsq(mtx1,mtx2)[0]
    
    roots, _ =  np.linalg.eig(QZ)    

    if mode in ['toeplitz']:roots = np.conj(roots)
    return roots

#-------------------------------------------------------------
def matrix_pencil_cov(x, order, mode, tls_rank = None):
    ''' 
    Matrix Pencil Method (MPM) 
        for of the decay signals model
        parameters estimation for 
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

    Returns
    ------------
    * roots: 1d ndarray,
        signal parameters in roots form.
    
    Notes
    ----------------
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
    x = np.asarray(x)
    N = x.shape[0]

    mtx = operators.covariance_matrix(x,
                                      mode=mode,
                                      lags = order+1)
    mtx1 = mtx[:-1,:]
    mtx2 = mtx[1:,:]
    
    if tls_rank:
        _tls_rank = min(max(order,tls_rank),N)
        mtx1, mtx2 \
            = operators.tls_turnication(mtx1, mtx2, _tls_rank)
    
    QZ=scipy.linalg.lstsq(mtx1,mtx2)[0]
    roots, _ =  np.linalg.eig(QZ)

    if mode in ['toeplitz']:roots = np.conj(roots)
    return roots
#-----------------------------------------
def kernel_matrix_pencil(x, order, mode, kernel = 'rbf', 
                         kpar=1, tls_rank = None):
    ''' 
    Matrix Pencil Method (MPM) 
        for of the decay signals model
        parameters estimation for 
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

    Returns
    ------------
    * roots: 1d ndarray,
        signal parameters in roots form.
    
    Notes
    ----------------
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
    x = np.asarray(x)
    N = x.shape[0]

    mtx = operators.kernel_matrix(x,
                                  mode=mode, 
                                  kernel=kernel,
                                  kpar=kpar,
                                  lags = order+1,)
    
    mtx1 = mtx[:-1,:]
    mtx2 = mtx[1:,:]

    mtx1, mtx2 \
        = operators.tls_turnication(mtx1, mtx2, order)
    
    QZ=scipy.linalg.lstsq(mtx1,mtx2)[0]
    
    roots, _ =  np.linalg.eig(QZ)
    return roots

# def matrix_pencil(x, order, mode, n_psd = None):
#     ''' 
#     Matrix Pencil Method (MPM) for decay signals model.
    
#     Parameters
#     --------------
#     * x: input 1d ndarray.    
#     * order: the model order.    
#     * mode:  mode for autoregression problem solving.    
#     * n_psd: length of reconstructed signal (x.size as defaults).

#     Returns
#     --------------
#     * Array with dementions order x x.size - signal components.
    
#     Notes
#     ----------------
#     * See prony as alternative for this function.
#     * MPM is calculated for each component as follows:
#      ..math::    
#      s_p(n)=[v_N^T * λ_p]^(-1)*x(n)*λ_p^n,
#      where 
#      * s_p(n) is the p-th component of decomposition;
#      * v^N is the Vandermonde matrix operator of degrees from 0 to N-1; 
#      * λ_p is the p-th eigenvalue of matrix r_1^(-1)(n)*r_2(n),  
#         where r_1(n) and r_2(n) are rows from 0 to P-1 (r_1(n)) 
#         and rows from 1 to P (r_2(n)) 
#        of the transported lags_matrix of x(n) with order P+1.  

#     Examples
#     -----------
    
    
#     References
#     -----------
#     [1a] A. Fernandez-Rodriguez, L. de Santiago, 
#     M.E.L. Guillen, et al.,"Coding Prony’s method in 
#     MATLAB and applying it to biomedical signal filtering", 
#         BMC Bioinformatics, 19, 451 (2018).
        
#     [1b]  https://bmcbioinformatics.biomedcentral.com/
#     articles/10.1186/s12859-018-2473-y 
     
#     '''
#     x = np.asarray(x)
#     N = x.shape[0]

#     if (n_psd is None): n_psd = N
    
#     #TODO: elumenate transpose
#     trj = operators.lags_matrix(x,
#                                 mode=mode,
#                                 lags = order+1)
#     trj1 = trj[:-1,:]
#     trj2 = trj[1:,:]
#     QZ = np.dot(np.linalg.pinv(trj1),trj2)
#     D, _ =  np.linalg.eig(QZ)
#     #     fs = N
#     #     dumps = np.log(np.abs(D))*fs
    
# #     freqs = np.angle(D)/2/np.pi

#     #TODO: replace vandermonde on fft - to increas stability.
#     v = ut.vander(D, N, True).T
#     h = np.dot(np.linalg.pinv(v),x)
    
#     #     amps  = np.abs(h)
#     #     thets = np.angle(h)    
#     out = np.zeros((order,N),dtype = x.dtype)    
#     k = np.arange(N)
#     #TODO: replace on fft to increas stability
#     for i in np.arange(order):
#         out[i,:] = h[i]*np.power(D[i],k) 
   
#     return out

# #-------------------------------------------------------------
# def matrix_pencil_tls(x, order, mode = 'toeplitz', tls_rank = 0, n_psd = None):
#     ''' 
#     Matrix pencil method using 
#         total least square for decay signals model.

#     Parameters
#     --------------
#     * x: input 1d ndarray.    
#     * order: the model order.    
#     * mode:  mode for autoregression problem solving.   
#     * tls_rank: If tls_rank>order, than tls with rank value 
#         will be perfermend, else tls with rank = order will be 
#         perfermed.
#     * n_psd: length of reconstructed signal (x.size as defaults).
     
#     Returns
#     --------------
#     * Array with dementions order x x.size - signal components.
    
#     Notes
#     ------------
#     * Mode toeplitz is reccomended.
#     * MP is calculated for each component as follows:
#      ..math::    
#       s_p(n)=[v_N^T * λ_p]^(-1)*x(n)*λ_p^n,
    
#      where 
#      * s_p(n) is the p-th component of decomposition;
#      * v^N is the Vandermonde matrix operator of degrees from 
#        0 to N-1; 
#      * λ_p is the p-th eigenvalue of matrix r_1^(-1)(n)*r_2(n),  
#        where r_1(n) and r_2(n) are rows from 0 to P-1 (r_1(n))
#        and rows from 1 to P (r_2(n)) of the transported
#        lags_matrix of x(n) with order P+1.  

#     See also
#     -------------------
#     matrix_pencil
#     prony
    
#     Examples
#     ----------------------
    
    
#     References
#     -------------------------
#     [1a] A. Fernandez-Rodriguez, L. de Santiago,
#     M.E.L. Guillen, et al., "Coding Prony’s method 
#     in MATLAB and applying it to biomedical signal filtering", 
#         BMC Bioinformatics, 19, 451 (2018).
        
#     [1b] https://bmcbioinformatics.biomedcentral.com/
#     articles/10.1186/s12859-018-2473-y 
     
#     '''
#     x = np.asarray(x)
#     N = x.shape[0]

#     if (n_psd is None): n_psd = N
        
#     if (tls_rank is None): tls_rank = order
    
#     #TODO: elumenate transpose
#     trj = operators.lags_matrix(x,mcolumns = order+1,mode=mode)
    
#     trj1 = trj[:-1,:]
#     trj2 = trj[1:,:]
    
#     trj1, trj2 =\
#         ut.tls_turnication(trj1,trj2, tls_rank=max(tls_rank,order))
    
#     QZ = np.dot(np.linalg.pinv(trj1),trj2)
    
#     D, _ =  np.linalg.eig(QZ)
#     #     fs = N
#     #     dumps = np.log(np.abs(D))*fs
#     #     freqs = fs*np.angle(D)/2/np.pi

#     #TODO: replace vandermonde on fft - to increas stability.
#     v = np.vander(D, N, True).T
#     h = np.dot(np.linalg.pinv(v),x)
    
#     #     amps  = np.abs(h)
#     #     thets = np.angle(h)    
#     out = np.zeros((order,N),dtype = x.dtype)    
#     k = np.arange(N)
#     #TODO: replace on fft to increas stability
#     for i in np.arange(order):
#         out[i,:] = h[i]*np.power(D[i],k) 
   
#     return out

# #-------------------------------------------------------------
# def matrix_pencil_rbf(x, order, mode, gamma=1, n_psd = None):
#     ''' 
#      RBF Matrix pencil method for decay signals model.
    
#     Parameters
#     ---------------
#     * x: input 1d ndarray.    
#     * order: the model order.    
#     * mode:  mode for autoregression problem solving.  
#     * gamma: coefficient of kernel: K = exp(-gamma*[X[1:]-X1[:-1]]^2).
#     * n_psd: length of reconstructed signal (x.size as defaults).
    
#     Returns
#     --------------
#     * Array with dementions order x x.size - signal components.
    
#     Notes
#     --------------
#     * See prony as alternative for this function.
#     * MP is calculated for each component as follows:
#        ..math::    
#        s_p(n)=[v_N^T * λ_p]^(-1)*x(n)*λ_p^n,
    
#        where 
#        * s_p(n) is the p-th component of decomposition;
#        * v^N is the Vandermonde matrix operator of degrees 
#            from 0 to N-1; 
#        * λ_p is the p-th eigenvalue of matrix r_1^(-1)(n)*r_2(n),  
#           where r_1(n) and r_2(n) are rows from 0 to P-1 (r_1(n)) 
#           and rows from 1 to P (r_2(n)) 
#           of the transported lags_matrix of x(n) with order P+1.  

    
#     Examples
#     ------------
    
    
#     References
#     -----------------
#     [1a] A. Fernandez-Rodriguez, L. de Santiago, 
#     M.E.L. Guillen, et al.,"Coding Prony’s method 
#     in MATLAB and applying it to biomedical signal filtering", 
#         BMC Bioinformatics, 19, 451 (2018).
        
#     [1b]  https://bmcbioinformatics.biomedcentral.com/
#     articles/10.1186/s12859-018-2473-y 
     
#     '''
#     x = np.asarray(x)
#     N = x.shape[0]

#     if (n_psd is None): n_psd = N
    
#     #TODO: elumenate transpose
#     trj = operators.kernel_matrix(x,
#                                   mode=mode, 
#                                   kernel=kernel,
#                                   kpar=kpar,
#                                   lags = order+1,)
#     trj1 = trj[:-1,:]
#     trj2 = trj[1:,:]
#     trj1, trj2 \
#         = ut.tls_turnication(trj1,trj2, tls_rank=order)
    
#     QZ = np.dot(np.linalg.pinv(trj1),trj2)
#     D, _ =  np.linalg.eig(QZ)
#     #     fs = N
#     #     dumps = np.log(np.abs(D))*fs
#     #     freqs = fs*np.angle(D)/2/np.pi

#     #TODO: replace vandermonde on fft - to increas stability.
#     v = mp.vander(D, N, True).T
#     h = np.dot(np.linalg.pinv(v),x)
    
#     #     amps  = np.abs(h)
#     #     thets = np.angle(h)    
#     out = np.zeros((order,N),dtype = x.dtype)    
#     k = np.arange(N)
#     #TODO: replace on fft to increas stability
#     for i in np.arange(order):
#         out[i,:] = h[i]*np.power(D[i],k) 
   
#     return out


#--------------------------------------
# def matrix_pencil_cov(x, order, mode, n_psd = None, tls_rank=None):
#     ''' 
#     Matrix pencil method for covariance signals model.
    
#     Parameters
#     -------------
#     * x: input 1d ndarray.    
#     * order: the model order.    
#     * mode:  mode for autoregression problem solving.  
#     * n_psd: length of reconstructed signal (x.size as defaults).
#     * tls_rank: if not None, than tls turnication with rank
#         max(tls_rank,order) will be perfermed.
        
#     Returns
#     ----------
#     * Array with dementions order x x.size - signal components.
    
#     Notes
#     ---------
#     * See prony as alternative for this function.
#     * Pseudo-spectrum could be reconstructed as:
#          recsig = matrix_pencil(x, order, n_psd = None)
#          psd = ut.afft(np.sum(recsig,axis=1))
#     * Chosed components could be recinstructed as:
#          recsig = matrix_pencil(x, order, n_psd = None)
#          indexes = [0,1,2]
#          signal = np.sum(recsig[:,indexes],axis=1)         
    
#     * MP is calculated for each component as follows:
#       ..math::    
#       s_p(n)=[v_N^T * λ_p]^(-1)*x(n)*λ_p^n,
    
#       where 
#       * s_p(n) is the p-th component of decomposition;
#       * v^N is the Vandermonde matrix operator of degrees from 0 to N-1; 
#       * λ_p is the p-th eigenvalue of matrix r_1^(-1)(n)*r_2(n),  
#         where r_1(n) and r_2(n) are rows from 0 to P-1 (r_1(n)) 
#         and rows from 1 to P (r_2(n)) 
#         of the transported lags_matrix of x(n) with order P+1.  

    
#     Examples
#     -------------
    
    
#     References
#     -------------
#     [1a] A. Fernandez-Rodriguez, L. de Santiago, 
#     M.E.L. Guillen, et al., "Coding Prony’s method in MATLAB 
#     and applying it to biomedical signal filtering", 
#         BMC Bioinformatics, 19, 451 (2018).
        
#     [1b]  https://bmcbioinformatics.biomedcentral.com/
#     articles/10.1186/s12859-018-2473-y 
     
#     '''
#     x = np.asarray(x)
#     N = x.shape[0]

#     if (n_psd is None): n_psd = N
    
#     #TODO: elumenate transpose
#     trj = operators.covariance_matrix(x,
#                                       mode=mode,
#                                       lags = order+1)
#     trj1 = trj[:-1,:]
#     trj2 = trj[1:,:]
    
#     if tls_rank is not None: 
#         trj1, trj2 \
#         = ut.tls_turnication(trj1,trj2, tls_rank=max(tls_rank,order))
    
#     QZ = np.dot(np.linalg.pinv(trj1),trj2)
#     D, _ =  np.linalg.eig(QZ)
#     #     fs = N
#     #     dumps = np.log(np.abs(D))*fs
#     #     freqs = fs*np.angle(D)/2/np.pi

#     #TODO: replace vandermonde on fft - to increas stability.
# #     v = matrix.vandermonde(D, N).T
#     v = np.vander(D,N,True).T
#     h = np.dot(np.linalg.pinv(v),x)
    
#     #     amps  = np.abs(h)
#     #     thets = np.angle(h)    
#     out = np.zeros((order,N),dtype = x.dtype)    
#     k = np.arange(N)
#     #TODO: replace on fft to increas stability
#     for i in np.arange(order):
#         out[i,:] = h[i]*np.power(D[i],k) 
   
#     return out

# #-------------------------------------------------------------
# def prony(x,order, mode = 'covar', n_psd = None):
#     ''' 
#     Prony-modifyed method for decay signals model reconstruction.
    
#     Parameters
#     ---------------------------------------------------
#     * x: input 1d ndarray.
#     * order: the model order.
#     * mode:  mode for autoregression problem solving.
#     * n_psd: length of reconstructed signal (x.size as defaults).

#     Returns
#     ----------------------------------------------
#     * array with dementions x.size x order - signal components.
    
#     Notes
#     ----------------------------------------------
#     * See matrix.lags_matrix modes for more mode experiments.
#     * See matrix_pencil as alternative for this function.
#     * Pseudo-spectrum could be reconstructed as:
#        recsig = prony(x, order, n_psd = None)
#        psd = ut.afft(np.sum(recsig,axis=1))      
#     * Chosed components could be reconstructed as:
#          recsig = prony(x, order, n_psd = None)
#          indexes = [0,1,2]
#          signal = np.sum(recsig[:,indexes],axis=1)         
    
    
#     Examples
#     ----------------------------------------------------------
    
#     References    
#     ----------------------------------------------------
#     [1a] A. Fernandez-Rodriguez, L. de Santiago, M.E.L. Guillen, et al., 
#         "Coding Prony’s method in MATLAB and applying it to biomedical signal filtering", 
#         BMC Bioinformatics, 19, 451 (2018).
#     [1b]  https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2473-y    
#     [2] S.L. Marple, Digital spectral analysis with applications. – New-York: Present-Hall, 1986.
   
#     '''
#     x = np.asarray(x)
#     N = x.shape[0]

#     if (n_psd is None): n_psd = N
        
#     #TODO: replace on arma.covar  ?  
#     a,_ = spectrum.lsar(x, order, mode=mode,n_psd=-1)
    
#     roots = np.roots(a) #dumps=np.log(np.abs(roots))*fs; freqs = fs*np.angle(roots)/2/np.pi
    
#     #TODO: replace on fft to increas stability
#     v  = matrix.vandermonde(roots, N).T 
    
#     #FOR CLASSIC SOLUTION vandermonde(r, order).T
#     # x1 = x[:order] #FOR LS SOLUTION x1 = x
    
#     # TODO: look for fast way to solve this equation (based on vandermonde properties)
#     h  = np.dot(np.linalg.pinv(v),x)# amps  = np.abs(h); thets = np.angle(h)    
#     out = np.zeros((order,N),dtype = x.dtype)
    
#     #TODO: replace on fft to increas stability
#     k = np.arange(N)
#     for i in np.arange(order):
#         out[i,:] = h[i]*np.power(roots[i],k) 
   
#     return out


# def prony(x, order):
#     '''
    
#     '''
    
#     """ Prony decomposition of signal.
#     https://dsplab.readthedocs.io/en/latest/prony.html
#     Parameters
#     ----------
#     xdata: array_like
#         Signal values.
#     ncomp: integer
#         Number of components. 2*ncomp must be less tham length of xdata.

#     Returns
#     -------
#     : np.array
#         Mu-values.
#     : np.array
#         C-values.
#     : np.array
#         Components.
#     """
#     x = np.asarray(x)
#     ncomp = order
#     N = len(x)
    
#     if 2*ncomp > N:
#         return None
    
#     d = np.array(xdata[ncomp:])
    
#     D = []
    
#     for i in range(ncomp, N):
#         D_row = []
#         for j in range(0, ncomp):
#             D_row.append(xdata[i-j-1])
#         D.append(np.array(D_row))
    
#     D = np.array(D)
    
#     a = np.linalg.lstsq(D, d)[0]

#     p = np.array([1] + [-ai for ai in a])
    
#     ms = np.roots(p)

#     d = np.array(xdata[:N])
#     D = []
    
#     for i in range(0, N):
#         D_row = []
#         for j in range(0, ncomp):
#             D_row.append(ms[j]**i)
#         D.append(np.array(D_row))
    
#     D = np.array(D)

#     cs = np.linalg.lstsq(D, d)[0]

    
#     es = []
    
#     for i in range(0, ncomp):
#         e = [cs[i]*(ms[i]**k) for k in range(0, N)]
#         es.append(np.array(e))
#     es = np.array(es).T

#     return ms, cs, es
