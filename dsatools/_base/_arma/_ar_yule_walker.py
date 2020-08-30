import numpy as np
import scipy  

__all__ = ['ar_yule_walker']

from ... import operators
from ... import utilits as ut

#------------------------------------------------------------------
def ar_yule_walker(x, order, mode='straight', unbias = False):    
    '''    
    Yule-Walker method for AutoRegressive (AR) model approximation.

    Parameters
    --------------
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
    * If mode 'full', 
        array of doubled size will be returned.
    * Yule-Walker method:
      ..math::
      r1 = -toeplitz(r_0,...r_{p-1})A^T,

      where 
      * r  = [r_0,r_1,...r_{p-1}] 
          - autocorrelation coefficients;
      * r1 = [r_1,r_1,...r_p]^T   
          -set of required approximation results;
      * p  is the model order;
      * toeplitz is the operator 
          for obtaining toeplitz matrix from vector;
      * A={a_1,...,a_p} 
          are the approcimation coefficients 
                                      of autoregression model 
        with the order p, in the following form:
        r_m = \sum_{k=1}^p {a_k r_{m-k}}+\sigma^2
        where \sigma^2 
            is the ewsidual noise, which can be calulated as
        \sigma = sqrt(r_0 + \sum(r_1^* \cdot a)).
    
       
    Examples
    --------------
    
    References
    -------------
    [1a] P. Stoica, R.L. 
        Moses, Spectral analysis of signals 
        - New-York: Present-Hall, 2005.
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
    ----------
    toeplitz; 
    covar; 
    levenson_durbin; 
    burg.
    
    '''    
    x = np.asarray(x)
    N = x.shape[0]
    
    r  = operators.correlation(x, y=None, mode=mode,
                            take_mean=False, unbias=unbias)  
    
    rn = scipy.linalg.toeplitz(r[:order]) 

    r1 = r[1:order + 1]    

#     a  = np.dot(np.linalg.inv(rn),r1)
    a = scipy.linalg.lstsq(rn,r1)[0]
    noise_var = np.real(r[0] - np.dot(np.conj(r1),a))

    a = np.append(1,-a)

    return a, noise_var

