import numpy as np
import scipy  

__all__ = ['arburg','arburg_cov']

from ... import operators 

#-----------------------------------------
def ar_burg(x, order):    
    '''
    The autoregressive model approximation, based 
        on the Burg itterative approximation 
        that minimize forece and back variance of the model.

    Parametres
    ----------------------
    * x:  1d ndarray. 
    * order: int.
        the autoregressive model (pole model) 
        order of the desired model. 

    Returns
    ------------------
    * a: 1d ndarray (complex (or float)),
        autoregression coefficients. 
    * noise_variace: complex (or float), 
        variance of model residulas.

    
    Examples
    -----------------
    
    See also
    -----------------
    yule_walker,
    lsar,
    levenson_durbin,
    covar, 
    mcovar.    
    
    References
    ------------------
    [1a] P. Stoica, R.L. Moses, Spectral analysis of signals 
                        - New-York: Present-Hall, 2005.
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
            - Dr.Moses Spectral Analysis of Signals: Resource Page
    [2a] M.H. Hayes. Statistical Digital Signal Processing 
                            and Modeling, John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
            2183-statistical-digital-signal-processing-and-modeling
    [3]  S.L. Marple, Digital spectral analysis with applications. 
                    – New-York: Present-Hall, 1986.
    

    
    '''    
    x = np.array(x)
    N = x.shape[0]

    a  = np.zeros(order, dtype=np.complex)
    k  = np.zeros(order, dtype=np.complex) #need only for variance
    
    ef = x.astype(np.complex)
    eb = x.astype(np.complex)
    
    for i in np.arange(order):
        
        #TODO: eluminate k (reduce array)
        efp = ef[i+1:]
        ebp = eb[i:-1]
        
        num = np.sum(efp*np.conj(ebp))        
        den = np.sum(ebp*np.conj(ebp))+np.sum(efp*np.conj(efp))
        k[i] = -2*num/den 
       
        a[i] = k[i]
        if i > 0: a[:i] = a[:i]+ k[i]*np.conj(a[i-1::-1])
        
        tmp1 = ef[i+1:] + k[i]*eb[i:-1]
        tmp2 = eb[i:-1] + np.conj(k[i])*ef[i+1:]
        ef[i+1:] = tmp1
        eb[i+1:] = tmp2
    
    a = np.append(1,a[:])    
    
    var = 1

    var = np.sum(x*np.conj(x))/N
    for i in np.arange(order):var = var*(1-k[i]*np.conj(k[i]))
    return a, var 

#--------------------------------------
def ar_burg_covar(x, order, mode='straight', unbias = False):    
    '''
    The autoregressive model approximation, based 
            on the Burg itterative approximation 
            that minimize forece and back variance of the 
            models autocorrelation function.

    Parametres
    ----------------------
    * x: 1d ndarray,
        1-d input ndarray;
    * order: int,
        is the order of the desired model;  
    * mode: string,
        mode of correlation function, 
        mode = {full, same, straight}.
    * unbias: bool, 
        if True, unbiased covariance function will be taken.
      
    
    Returns
    ------------------
    * a: 1d ndarray (complex (or float)),
        autoregression coefficients. 
    * noise_variace: complex (or float), 
        variance of model residulas.
              
    Notes
    ---------
    * If mode 'full', array of doubled size will be returned.
    
    Examples
    -----------------
        
    See also
    -----------------
    burg,    
    lsar,
    arma_tools,
    arma (module).    
    
    References
    ------------------
    [1a] P. Stoica, R.L. Moses, Spectral analysis of signals 
                        - New-York: Present-Hall, 2005.
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
            - Dr.Moses Spectral Analysis of Signals: Resource Page
    [2a] M.H. Hayes. Statistical Digital Signal Processing 
                            and Modeling, John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
            2183-statistical-digital-signal-processing-and-modeling
    [3]  S.L. Marple, Digital spectral analysis with applications. 
                    – New-York: Present-Hall, 1986.

    '''    
    r = operators.correlation(x,mode=mode,unbias=unbias)
    return ar_burg(r,order=order)
    