import numpy as np
import scipy  

from ... import operators 

__all__ = ['hoyw']

def ar_hoyw(x, order, mode='full', lags=None):
    '''
    Estimation of the pseudo-spectrum based on 
      the High order Yule-Walker (HOYW) autoregression method.

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

    Returns
    ------------------
    * a: 1d ndarray (complex (or float)),
        autoregression coefficients. 
    * noise_variace: complex (or float), 
        variance of model residulas.
    
    Notes
    -----------
    * HOYW pseudospectrum is calculated as:
      ..math::    
      P(z)= 1/(a(z)G(z)a^H),
      where  
      * a = exp(-j2pi*f/fs); 
      * G(z) is the covariance matrix approximation 
      of rank = order.
    
    
    References
    ------------------------
    [1] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [2] http://www2.ece.ohio-state.edu/~randy/SAtext/
        - Dr.Moses Spectral Analysis of Signals: Resource Page
        
    Examples
    ----------------
    

    '''    
    x = np.array(x)
    N = x.shape[0]

    if(lags is None): lags = N//2

    R = operators.covariance_matrix(x, 
                                    lags=lags, 
                                    mode=mode)

    es,ev = np.linalg.eig(R)

    
    D = np.diag(1/es[:order])
    V1 = ev[:,:order]
    
    #TODO: make in form of AR
    # compute the estimate of the a (AR) coefficients
    a = np.inner(V1 @ D @ np.conj(V1).T , R[0,:])
    a = a[::-1]
#     a = np.append(1,a)
    err = 1
    return a,err
 
    
    
# def hoyw(x, order, mode='full', lags=None, n_psd=None, n_predict=1):
#     '''
#     Estimation of the pseudo-spectrum based on 
#       the High order Yule-Walker (HOWY) autoregression method.

#     Parameters
#     --------------------------
#     * x:      Input 1d signal.
#     * order:  Order of the model (rank of matrix approximation).     
#     * mode:   The mode of correlation function 
#                 (full as default), also aviliable: caterpillar(traj),
#                 covar, mcovar, toeplitz and hankel modes.
#     * lags: Number of lags in the correlation matrix 
#                         (Nlags =x.shape[0]//2 if None).
#     * n_psd:  Length of psceudo-spectrum  (Npsd = x.shape[0] if None),
#         if n_psd <0, then ar coefficients will be returned.
                    
#     * n_predict: Length of space to solve AR equation 
#                                 (n_pred = 1 by default). 
  
#     Returns
#     --------------------------
#     * pseudo-spectrum - 1d ndarray 
    
#     Notes
#     -------------------------
#     * In some modes psd more correct it n_pred>1.
#     * HOYW pseudospectrum is calculated as:
#       ..math::    
#       P(z)= 1/(a(z)G(z)a^H),
#       where  
#       * a = exp(-j2pi*f/fs); 
#       * G(z) is the covariance matrix approximation of rank = order.
    
    
#     References
#     ------------------------
#     [1] Stoica, Petre, and Randolph L. Moses. 
#         "Spectral analysis of signals." (2005).
#     [2] http://www2.ece.ohio-state.edu/~randy/SAtext/
#         - Dr.Moses Spectral Analysis of Signals: Resource Page
#     [3] M.H. Hayes. 
#         Statistical Digital Signal Processing and Modeling, 
#         John Wiley & Sons, 1996.
#     [4] https://www.mathworks.com/matlabcentral/fileexchange/2183
#         -statistical-digital-signal-processing-and-modeling
    
#     Examples
#     ----------------
    

#     '''    
#     x = np.array(x)
#     N = x.shape[0]
    
#     if(n_psd is None):  n_psd  = int(N)
#     if(lags is None): lags = N//2

#     R = matrix.covariance_matrix(x, lags=lags, 
#                                  mode=mode, fb=False)

#     es,ev = np.linalg.eig(R)
#     D = np.diag(1/es[:order])

#     # find the approximation matrix 
#     D  = np.matrix(D)
#     V1 = np.matrix(ev[:,:order])

#     #TODO: make in form of AR
#     # compute the estimate of the a (AR) coefficients
#     a = -V1 * D * V1.H * np.matrix(R[:n_predict,:]).T
    
#     a = np.append(1,a)
    
#     if n_psd<0:
#         err=1
#         return a,err
    
#     psd = np.zeros(n_psd)    
#     n = np.arange(lags)
    
#     for i in np.arange(n_psd):       
#         freqvect  = np.exp(2j*np.pi*n*i/n_psd)
#         psd[i] = np.linalg.norm(freqvect*a)

#     return 1/psd