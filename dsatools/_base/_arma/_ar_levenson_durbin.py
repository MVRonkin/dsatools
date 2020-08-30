import numpy as np
import scipy  


from ... import operators
#------------------------------------------ 
def ar_levenson_durbin(x, order,mode='same',unbias = False):
    '''
    The autoregressive model approximation, 
        based on the Levenson-Dubrin itterative method 
        for solution toeplitz matrix equations.
   
    Parameters
    -------------------
    * x: is 1-d input ndarray.
    * order: int,
        is the order of the desired model. 
    * mode: string,
        mode of correlation function, 
        mode = {full, same, straight}.
    * unbias: bool,
        if True, the unbiased autocorrleation 
         function will be taken.

    Returns
    ---------------------
    * a: 1d ndarray,
        autoregression coefficients,  
    * noise_variace: float or complex,
        variance of model residulas.
       
    See also 
    ------------
    yule_walker, 
    lsar,
    covar, 
    burg.
    
    Examples
    ----------- 
    
    References
    --------------------
    [1a] Stoica, Petre, and Randolph L. Moses. 
            "Spectral analysis of signals." (2005).
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
    - Dr.Moses Spectral Analysis of Signals: Resource Page
    [2a] M.H. Hayes. Statistical Digital Signal Processing 
                        and Modeling, John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling
    [3]  S.L. Marple, Digital spectral analysis with applications. 
                                    – New-York: Present-Hall, 1986.
    

    
    '''    
    x = np.asarray(x)
    N = x.shape[0]

    r  = operators.correlation(x,y=None,mode=mode,
                        take_mean=False,unbias=unbias)    
    
    a = np.zeros((order,), x.dtype)
    var  = r[0] - (r[1] *np.conj(r[1]))/r[0]
    a[0] = -r[1] / r[0]

    for i in np.arange(1,order):        
        k   = -(r[i+1] + np.sum(a[:i]*r[i:0:-1]))/var #r[i:0:-1]=np.flipud(r[1:i+1])
        var = var*(1-(k *np.conj(k)))
        
        # same as a[:i+1] = [a,0] + k[a~,1] in Stoic
        a[:i] = a[:i] + k*np.conj(a[i-1::-1])      
        a[i]  = k

    # here is sign "-" is already taken 
    a = np.append(1,a)    

    return a, var    
    



