import numpy as np
import scipy  

from ... import operators
from ... import utilits as ut

#------------------------------------------------------------------
__all__ = ['ar_ls']

def ar_ls(x,order, mode='full'):
    '''    
    Least-square autoregression method based on the lags matrixes.
   
    Parameters
    -------------
    * x: 1-d ndarray,
        input signal.
    * order: int,
        is the order of the desired model.  
    * mode: string,
        mode of lags matrix for ls problem solution,
        modes = {full,  prew,postw, covar}. 

    Returns
    ---------------------
    * a: 1d ndarray,
        autoregression coefficients,  
    * noise_variace: float or complex,
        variance of model residulas.
    
    Notes
    --------
    * In Marple terminology function describe both covariance method 
             (mode = 'covar')
             and modified covarinace method (mode = 'mcovar').
    * In Hayes terminology function describe 
        autocorrelation and covariation methods.
    * In Stoica terminology function has name lsar.

    * Autoregression equation:
      ..math::                
      r  = data_matrix_{p+1}(x),
      a  = [1,-r[:,1:]^(-1)r[:,0]],        
      where 
        * r = [[r,0,...,0]^T,[0,r...0]^T,...,[0,...0,r]^T] 
            - is the lags matrix, 
              (i.e. so called convolution aor even covariance matrix);               
        * p  is the model order;              
        * a={a_1,...,a_p} 
            are the approcimation coefficients of the autoregression 
          model with the order p in the following form:                
          ..math::   
          r_m = \sum_{k=1}^p {a_k r_{m-k}}+\sigma^2            
          where 
          * \sigma^2 is the ewsidual noise, which can be calulated as
          ..math::        
          \sigma = sqrt(r_0 + r1^Tr \cdot n \cdot r).
    
    Examples
    ------------
    
    References
    -----------
    [1a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page
    [2a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling
    [3]  S.L. Marple, Digital spectral analysis with applications. 
        – New-York: Present-Hall, 1986.
    
    See also
    ------------
    data_matrix; 
    yule_walker; 
    levenson_durbin; 
    burg; 
    covar; 
    mcovar.
    
    '''
    x = np.asarray(x)
    N = x.shape[0] 

    r = operators.lags_matrix(x, lags = order+1,mode = mode)
    r1 = r[:,0]
    rn = r[:,1:]
#     a = np.dot(np.linalg.pinv(-rn),r1) #same as scipy.linalg.lstsq(rn, r1)
    a = scipy.linalg.lstsq(rn,r1)[0]
    a = np.append(1,-a)

    err = np.abs(np.dot(np.dot(r[:,0].transpose(),r),a))
    return a, err
    
    
# #------------------------------------------------------------------
# def mcov(x, order, n_psd = None):
#     '''    
#     Alternative implementation of modified covariance method.
#     Strangely some-times works more stable.
#     Hayes
#     '''
#     x = np.asarray(x)
#     N = x.shape[0]
#     if n_psd is None: n_psd = N
# #     r = signals.matrix.toeplitz(x)[order:,:order+1]
#     r = matrix.datamtx(signal1,order+1,'covar')
#     R = np.dot(r,r.transpose().conj())
#     R1 = R[1:order+1,1:order+1]
#     R2 = matrix.backward_matrix(R[1:order+1,1:order+1],conj=False)
#     b1 = R[1:order+1,0]
#     b2 = np.flipud(R[:order,order+1])
#     rn = np.append(R1,R2,axis=1).T
#     r1 = np.hstack((b1,b2))

#     a = np.dot(np.linalg.pinv(-rn),r1)
    

#     a  = np.conj(np.append(1,a))
    

# #     err = R(1,:)*a+fliplr(R(p+1,:))*a;
   
#     err = 1

#     if(n_psd<1):
#         return a,err
#     else:
#         psd = arma2psd(a,1,np.abs(err),n_psd)
#         return psd    
    
    
# #------------------------------------------------------------------
# def covm(x, order):
#     '''    
#     Alternative implementation of modified covariance method.
#     Strangely some-times works more stable.
    
#     '''
#     x = np.asarray(x)    
#     N = x.shape[0]  
#     r = matrix.datamtx(x,order+1,'full')
    
#     rn = r[order:N-1,:order]
#     r1 = r[order+1:N,0]
#     a = np.dot(np.linalg.pinv(-rn),r1)
#     a = np.append(1,a)
#     err = np.abs(np.dot(np.dot(r[order+1:N,0].transpose(),r[order+1:N,:]),a))
#     return sarma2psd(a=a, b=1, err=np.abs(err), n_psd = N)  

# #------------------------------------------------------------------

# def mcovar(x,order, n_psd=None):
#     '''    
#     Alternative implementation of modified covariance method.
#     Strangely some-times works more stable.
    
#     '''
#     x = np.asarray(x)
#     N = x.shape[0] 
#     if (n_psd is None): n_psd = N

#     r = matrix.datamtx(x, mcolumns=order, mode='mcovar')
#     r = r[:-1]  
    
#     r1 = x[order:]
# #     if(mode=='mcovar'):
#     r1 = np.append(r1,np.conj(x[order-1:][::-1]))
        
#     r = np.matrix(r[:,:])
    
#     a  = np.dot(np.linalg.pinv(r),r1) #same as scipy.linalg.lstsq(-rc, r1)

#     a = np.append(1,-a)
#     noise_var = 1
    
#     #special case
#     if(n_psd<0):
#         noise_var = r[0] - np.dot(r1.H,np.dot(r,r1)) #TODO: check this
#         return a, noise_var
#     else:
#         return arma2psd(a=a, b=1, err=noise_var, n_psd = n_psd)

# #------------------------------------------------------------------
# def lscovar(x,order, n_psd=None):
#     '''    
#     Alternative implementation of covariance method.
#     Strangely some-times works more stable.
    
#     '''
#     x = np.asarray(x)
#     N = x.shape[0] 
#     if (n_psd is None): n_psd = N

#     r = matrix.datamtx(x, mcolumns=order, mode='covar')
#     r = r[:-1]  
    
#     r1 = x[order:]
# #     if(mode=='mcovar'):
# #     r1 = np.append(r1,np.conj(x[order-1:][::-1]))
        
#     r = np.matrix(r[:,:])
    
#     a  = np.dot(np.linalg.pinv(r),r1) #same as scipy.linalg.lstsq(-rc, r1)

#     a = np.append(1,-a)
#     noise_var = 1
    
#     #special case
#     if(n_psd<0):
#         noise_var = r[0] - np.dot(r1.H,np.dot(r,r1)) #TODO: check this
#         return a, noise_var
#     else:
#         return arma2psd(a=a, b=1, err=noise_var, n_psd = n_psd)

# #------------------------------------------------------------------
# def covar_covar(x, order, n_psd=None ):
#     '''    
#     Alternative implementation of covariance method.
#     Strangely some-times works more stable.
    
#     '''     
#     x = np.asarray(x)
#     N = x.shape[0]
#     if(n_psd is None):n_psd = N
    
#     r = correlation.correlation(x,mode='same', unbias=False)[N//2:]
#     a, noies = covar(r, order=order,mode='covar',n_psd=n_psd)
#     if(n_psd<1):
#         return a, noies
#     else:
#         return arma2psd(a=a, b=1, err=noise_var, n_psd = n_psd)
# #------------------------------------------------------------------    
# def acm(x, order):
#     x = np.asarray(x)    
#     N = x.shape[0]  
#     r = signals.matrix.datamtx(x,order+1,'full')
    
#     rn = r[:-1,:order]
#     r1 = r[1:,0]
#     print(r.shape, N+order, rn.shape,r1.shape)
#     a = np.dot(np.linalg.pinv(-rn),r1)
#     a = np.append(1,a)
#     err = np.abs(np.dot(np.dot(r[:,0].transpose(),r),a))
#     return signals.spectrum._arma._arma_tools.arma2psd(a=a, b=1, err=np.abs(err), n_psd = N)   