import numpy as np
import scipy  

__all__ = ['ar_min_norm', 'ar_kernel_minnorm']

from ... import operators

def ar_minnorm(x, order, mode='full',lags=None, signal_space = False ):
    '''  
    Estimation autoregression model coefficients based 
        on the minimum norm (max_entropy) algorithm.
    
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
    Returns
    ------------------
    * a: 1d ndarray (complex (or float)),
        autoregression coefficients. 
    * noise_variace: complex (or float), 
        variance of model residulas.

    Notes
    -----------
    * Use estimated parameters with arma_tools
        to eestiamte spectrum or signal parameters.
    
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
    x = np.asarray(x)
    N = x.shape[0]
    
    if lags is None: lags = N//2
    if(lags>N): raise ValueError('Nlags>x.shape[0]')
        
    R = operators.covariance_matrix(x, lags=lags, mode=mode)

    es,ev = np.linalg.eig(R)

    if (signal_space):
        signal_space=ev[:,:order]
        alpha = np.matrix(signal_space[0,:]) 
        Sbar  = signal_space[1:,:]

        Y = 1-alpha*alpha.H    
        a = np.append(1,- np.dot(Sbar,alpha.H)/Y)
    
    #TODO: As alternative noise subspace can be concidered
    # see Stoica SaS for more information.
    else:
        noise_space=ev[:,order:]
        betta = np.matrix(noise_space[0,:])
        Gbar  = noise_space[1:,:]
        betta_norm = betta*betta.H    
        a = np.append(1, np.dot(Gbar,betta.H)/betta_norm)
    a = np.conj(a)
    err = 1
    return a,err


#-----------------------------------------
def ar_kernel_minnorm(x, order, mode='full', 
                    kernel = 'linear', kpar = 1, 
                    lags=None, signal_space = False ):
    '''  
   Estimation autoregression model coefficients based 
        on the kernel minimum norm (max_entropy) algorithm.
    
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
    Returns
    ------------------
    * a: 1d ndarray (complex (or float)),
        autoregression coefficients. 
    * noise_variace: complex (or float), 
        variance of model residulas.

    Notes
    -----------
    * Use estimated parameters with arma_tools
        to eestiamte spectrum or signal parameters.
    
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
    x = np.asarray(x)
    N = x.shape[0]
    
    if lags is None: lags = N//2
    if(lags>N): raise ValueError('Nlags>x.shape[0]')
        
    R = operators.kernel_matrix(x,  
                                mode   = mode,
                                kernel = kernel,
                                kpar   = kpar,
                                lags   = lags)

    es,ev = np.linalg.eig(R)

    if (signal_space):
        signal_space=ev[:,:order]
        alpha = np.matrix(signal_space[0,:]) 
        Sbar  = signal_space[1:,:]

        Y = 1-alpha*alpha.H    
        a = np.append(1,- np.dot(Sbar,alpha.H)/Y)
    
    #TODO: As alternative noise subspace can be concidered
    # see Stoica SaS for more information.
    else:
        noise_space=ev[:,order:]
        betta = np.matrix(noise_space[0,:])
        Gbar  = noise_space[1:,:]
        betta_norm = betta*betta.H    
        a = np.append(1, np.dot(Gbar,betta.H)/betta_norm)
    a = np.conj(a)
    err=1
    return a,err
   


# #------------------------------------------------        
# def minimum_norm(x, order, mode='covar', fb=False, n_lags=None, n_psd=None):
#     '''  
#     Estimation of the pseudo-spectrum based 
#                     on the minimum norm algorithm. 
    
#     Parameters
#     -----------
#     * x:     Input 1d signal.
#     * order: Order of the model (number of valuable components, 
#                                         size of signal subspace). 
#     * mode:  The mode of correlation function ('full' by default)
#     * fb:    If ture force-back matrice 
#         (modified covariance matrix will be taken).
#     * n_psd: Length of pseudo-spectrum 
#                 (Npsd = x.shape[0] if None).
#     * n_lags: Number of lags in correlation function, 
#                             (x.shape[0]//2 by default).
    
#     Returns
#     -----------
#     * pseudo-spectrum 1d ndarray.
    
#     Notes
#     --------------
#     * See P. Stoic SaS for more information.                   
#     * For investigation the modes see correlation.covariance_matrix. 
#     * PSD  is calculated as follows
#       ..math::    
#       P(z)= 1/||a(z)^HG(z)||_2,    
#       where  
#       * a = exp(-j2pi*f/fs); 
#       * G(z) is  the eigenvector of the input signal 
#             correlation matrix with the minimal norm.
    
#     Example
#     ------------
    
#     References
#     ------------
#     [1a] Stoica, Petre, and Randolph L. Moses. 
#         "Spectral analysis of signals." (2005).
#     [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
#         - Dr.Moses Spectral Analysis of Signals: Resource Page
#     [2a] M.H. Hayes. 
#         Statistical Digital Signal Processing and Modeling, 
#         John Wiley & Sons, 1996.
#     [2b] https://www.mathworks.com/matlabcentral/fileexchange/
#         2183-statistical-digital-signal-processing-and-modeling
        
#     ''' 
#     x = np.asarray(x)
#     N = x.shape[0]
    
#     if n_lags is None: n_lags = N//2
#     if(n_psd is None): n_psd  = N 
#     if(n_lags>N): raise ValueError('Nlags>x.shape[0]')
        
#     R = matrix.covariance_matrix(x, lags=n_lags, mode=mode, fb=fb)

#     es,ev = np.linalg.eig(R)
    
#     in_signal_space = True
#     if (in_signal_space):
#         signal_space=ev[:,:order]
#         alpha = np.matrix(signal_space[0,:]) 
#         Sbar  = signal_space[1:,:]

#         Y = 1-alpha*alpha.H    
#         a = np.append(1,- np.dot(Sbar,alpha.H)/Y)
    
#     #TODO: As alternative noise subspace can be concidered
#     # see Stoica SaS for more information.
#     else:
#         noise_space=ev[:,order:]
#         betta = np.matrix(noise_space[0,:])
#         Gbar  = noise_space[1:,:]
#         betta_norm = betta*betta.H    
#         a = np.append(1, np.dot(Gbar,betta.H)/betta_norm)

    
#     psd=1/np.abs(np.fft.ifft(a,n_psd))
  
#     return psd
