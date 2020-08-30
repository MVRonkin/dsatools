import numpy as np
import scipy  

from ... import operators

from ._ar_ls import ar_ls

#------------------------------------------------------------------
__all__ = ['covar','mcovar']

def ar_cov(x, order, unbias = False, mode = 'covar', correlation_mode='same'):
    '''    
    Covariance method for autoregressive model 
        of covariation function approximation 
        based on the least-sqaure solution.

    Parameters
    ---------------------
    * x: id ndarray (complex or float),
        1-d input ndarray.
    * order: int,
        is the order of the desired model.  
    * unbias: bool,
        for tests, use True.    
    * mode: string,
        mode of autoregression problem solving,
        mode = {'full','covar','toeplitz','prew','postw','hankel'}.
    * correlation_mode: string,
        covariation function mode,
        correlation_mode = {full, straight, same}.

    Returns
    ------------------
    * a: 1d ndarray (complex (or float)),
        autoregression coefficients. 
    * noise_variace: complex (or float), 
        variance of model residulas.
              
    Notes
    ---------
    
    Examples
    ----------------
    
    See also: 
    ---------------
    yule_walker,
    levenson_durbin, 
    covar, 
    lsar.
    
    References
    -----------------
    [1a] M.H. Hayes. Statistical Digital Signal 
        Processing and Modeling, John Wiley & Sons, 1996.
    [1b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling
   
    '''
    x = np.asarray(x)
    N = x.shape[0]
    
    r = operators.correlation(x,mode=correlation_mode, unbias=unbias)#[N//2:]
    
    a, noies = ar_ls(r, order=order,mode=mode)

    return a, noies
    
   

#------------------------------------------------------------------   
    
def ar_mcov(x, order, mode = 'covar'):
    '''    
    Modified covariance method for autoregressive model 
        of covariation function approximation 
        based on the least-sqaure solution.

    
    Parametres
    ----------------------
    * x: 1d ndarray,
        1-d input ndarray;
    * order: int,
        is the order of the desired model.  
    * mode: string,
        mode of correlation function, 
        mode = {full, same, straight}.

    Returns
    ------------------
    * a: 1d ndarray (complex (or float)),
        autoregression coefficients. 
    * noise_variace: complex (or float), 
        variance of model residulas.
 
    Examples
    ----------------
    
    See also: 
    ---------------
    yule_walker,
    levenson_durbin, 
    covar, 
    lsar.
    
    References
    -----------------
    [1a] M.H. Hayes. Statistical Digital Signal 
        Processing and Modeling, John Wiley & Sons, 1996.
    [1b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling
   
   
    '''
    x = np.asarray(x)
    N = x.shape[0]
    
    #TODO: TRY TO REPLACE ON CORRELATION_MATRIX-MCOVAR
    r = operators.lags_matrix(x,mode=mode,  lags=order+1)    
    R = np.dot(r,r.transpose().conj())    
    
    R1 = R[1:order+1,1:order+1]    
    R2 = R1[::-1]#matrix.backward_matrix(R[1:order+1,1:order+1],conj=False)
    
    b1 = R[1:order+1,0]
    b2 = np.flipud(R[:order,order+1])
    rn = np.append(R1,R2,axis=1).T
    r1 = np.hstack((b1,b2))

    a = np.dot(np.linalg.pinv(-rn),r1)

    a = np.conj(np.append(1,a))
   
    err = 1

    return a,err
  
