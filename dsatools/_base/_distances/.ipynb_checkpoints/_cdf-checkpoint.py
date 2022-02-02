import numpy as np
import scipy

from ... import operators 

from ._distances import kl, minkowsky, entropy, _check_xy, _EPS_

#-----------------------------
def cdf_dist(x,y, p=1, smooth = 0, root = False):    
    ''' 
    Cummulitive-Distribution Function (CDF) distance.
    
    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray.
    * p: float,
        degree  parameter.
    * smooth: flaot,
        if not zero, than the entropy smoothed 
        value will be taken.
    * root: bool,
        if True, than root of value will be returned
        expect of the cased of p=0.
    
    Returns
    --------
    * cdf_dist: float.
    
    Notes
    ------------
    * Implementation is based on the SciPy implementation, 
      ..math::
      dist = sum(|cdf(x)-cfd(y)|^p*d(bins))
      where:
      * cdf() - is the cumulative-distribution function, 
        taken simultaneously for x and y; 
      * bins - is the corresponding bins samples 
        for both cdf(x) and cfd(y) together.
    * If smooth != 0:
      ..math::
      dist = dist-smooth*(bins*log(|bins|)).
    * Special case:  
      * Wasserstain or  earth moveing distance: p=1.
      * Energy distance: p=2.
  
    '''        
    x,y = _check_xy(x,y)
    out2 = 0   
    bins, P, Q  = operators.ecdf(x,y)
    
    deltas = operators.diff(bins)

    P = np.asarray(P, dtype=x.dtype)#+__EPSILON__
    Q = np.asarray(Q, dtype=y.dtype)#+__EPSILON__  
    deltas = np.asarray(deltas, dtype=np.complex)
    
    out = np.sum(np.power(np.abs(P - Q),p)*deltas)
    
    if root and (p!=0): 
        out = np.power(out,1/p)     

    if (smooth!=0) :       
        out2 = smooth*entropy(bins)
        
    return np.abs(out) + np.abs(out2)
#----------------------------------
def wasserstein(x,y):
    ''' 
    Wasserstein cummulitive-distribution function distance.
    
    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray.

    Returns
    --------
    * dist: float.
    
    Notes
    ------------
    * Implementation is based on the SciPy implementation, 
      ..math::
      dist = sum(|cdf(x)-cfd(y)|*d(bins))
      where:
        cdf() - is the cumulative-distribution function, 
        taken simultaneously for x and y; 
        bins - is the corresponding bins samples 
        for both cdf(x) and cfd(y) together.
    
    '''    
    return cdf_dist(x,y,p=1)
#----------------------------------
def energy(x,y, root = False):
    ''' 
    Energy cummulitive-distribution function distance.
    
    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray.
    * root: bool,
        if True, than root of value will be returned
        expect of the cased of p=0.
        
    Returns
    --------
    * dist: float.
    
    Notes
    ------------
    * Implementation is based on the SciPy implementation, 
      ..math::
      dist = sum(|cdf(x)-cfd(y)|^2*d(bins)),
      where:
        cdf() - is the cumulative-distribution function, 
        taken simultaneously for x and y; 
        bins - is the corresponding bins samples 
        for both cdf(x) and cfd(y) together.

    '''
    return cdf_dist(x,y,p=2, root = root)
#--------------------------------------------------
def minkowsky_cdf(x,y,p=2, root = False):
    ''' 
    Minkowsky cummulitive-distribution function distance.
    
    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray.
    * p: float,
        degree  parameter.
    * root: bool,
        if True, than root of value will be returned
        expect of the cased of p=0.
        
    Returns
    --------
    * dist: float.
    
    Notes
    ------------
    * Distance is calculated as, 
      ..math::
      dist = sum(|cdf(x)-cfd(y)|^p)
      where:
      cdf() - is the cumulative-distribution 
        function, taken simultaneously 
        for x and y.           
    '''
    x,y = _check_xy(x,y)
    
    _, P, Q  = operators.ecdf(x,y)

    out = np.sum(np.power(np.abs(P - Q),p))
    
    if root and p !=0: 
        out = np.power(out,1/p)     

    return np.abs(out) 

#----------------------------------
def kolmogorov_smirnov(x,y):
    ''' 
    Kolmogorov-Smirnov cummulitive-distribution 
      function distance.
    
    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray.
    
    Returns
    --------
    * dist: float.
    
    Notes
    ------------
    * Distance is calculated as, 
      ..math::
      dist = max(|cdf(x)-cfd(y)|)
      where:
      cdf() - is the cumulative-distribution 
        function, taken simultaneously 
        for x and y.           
    '''    
    x,y = _check_xy(x,y)
    
    _, P, Q = operators.ecdf(x,y)

    out = np.max(np.abs(P - Q))

    return np.abs(out) 

#----------------------------------------
def chisquare_cdf(x,y, p=2, root = False):
    ''' 
    Chi-square (or Chi-p_degree) 
        cummulitive-distribution function distance.
    
    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray.
    * p: float,
        degree  parameter.
    * root: bool,
        if True, than root of value will be returned
        expect of the cased of p=0. 
    
    Returns
    --------
    * dist: float.
    
    Notes
    ------------
    * Distance is calculated as, 
      ..math::
      dist = sum(|cdf(x)-cfd(y)|^p/cdf(x))
      where:
      cdf() - is the cumulative-distribution 
        function, taken simultaneously 
        for x and y.           
    '''
    x,y = _check_xy(x,y)

    _, P, Q  = operators.ecdf(x,y)

    out = np.sum(np.power(np.abs(P - Q),p)/(np.abs(P)+_EPS_))
    if root and p!=0: 
        out = np.power(out,1/p)     

    return np.abs(out) 
#----------------------------------
def cramer_vonmises(x,y, p=2, root = False):
    ''' 
    Cramer - von Mises 
        cummulitive-distribution function distance.
    
    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray.
    * p: float,
        degree  parameter.
    * root: bool,
        if True, than root of value will be returned
        expect of the cased of p=0. 
    
    Returns
    --------
    * dist: float.
    
    Notes
    ------------
    * Distance is calculated as, 
      ..math::
      dist = sum(|cdf(x)-cfd(y)|^p*d{cdf(y)})
      where:
      cdf() - is the cumulative-distribution 
        function, taken simultaneously 
        for x and y.
      d{} is the operator of differentiation.
        
    '''

    x,y = _check_xy(x,y)

    _, P, Q  = operators.ecdf(x,y)

    deltas = operators.diff(Q) 

    deltas = np.asarray(deltas, dtype=np.complex)    
    P = np.asarray(P, dtype=x.dtype)#+__EPSILON__
    Q = np.asarray(Q, dtype=y.dtype)#+__EPSILON__  

    out = np.sum(np.power(np.abs(P-Q),p)*deltas)
    if root and p!=0: 
        out = np.power(out,1/p)     

    return np.abs(out) 

#----------------------------------------------------------------------------------------
def anderson_darling(x,y, p=2, root = False):
    ''' 
    Anderson - Darling
        cummulitive-distribution function distance.
    
    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray.
    * p: float,
        degree  parameter.
    * root: bool,
        if True, than root of value will be returned
        expect of the cased of p=0. 
    
    Returns
    --------
    * dist: float.
    
    Notes
    ------------
    * Distance is calculated as, 
      ..math::
      dist = sum(|cdf(x)-cfd(y)|^p/(cfd(y)[1-cfd(y)])*d{cdf(y)})
      where:
      cdf() - is the cumulative-distribution 
        function, taken simultaneously 
        for x and y.
      d{} is the operator of differentiation.
        
    '''
    x,y = _check_xy(x,y)

    _, P, Q  = operators.ecdf(x,y)

    deltas = operators.diff(Q)/(Q*(1-Q)+_EPS_)
    deltas = np.asarray(deltas, dtype=np.complex)    
    P = np.asarray(P, dtype=x.dtype)#+__EPSILON__
    Q = np.asarray(Q, dtype=y.dtype)#+__EPSILON__  

    out = np.sum(np.power(np.abs(P-Q),p)*deltas)
    if root and p!=0: 
        out = np.power(out,1/p)     

    return np.abs(out) 

#----------------------------------------------------------------------------------------
def kl_cdf(x,y, a=-1, generalize = False):
    ''' 
    Kullback-Leibner (KL) divergence pf
        cummulitive-distribution functions.
    
    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray,
        if None, y = x.
    * a: float,
        alpha-Jenson-Shannon divergence parameter.
    * generalize, bool,
        if True, than generalized KL divergence 
        will be returned.

    Returns
    --------
    * divergence: float (or complex). 
    
    Notes
    ---------
    * KL divergence calculaed as
        dist = kl(P||Q) = sum(P*ln(|P/Q|)),
        where:
        P = cdf(x)*d{bins}; Q = cdf(y)*d{bins},
        where:
        * cdf() - is the cumulative-distribution 
          function, taken simultaneously 
          for x and y.
        * bins - is the corresponding bins samples 
          for both cdf(x) and cfd(y) together.
        * d{} is the operator of differentiation.
        
    * If (a>=0) it will be alpha-Jenson-Shannon Divergence: 
        dist = KL(P||a*Q+(1-a)*P)+KL(Q||a*P+(1-a)*Q)
      a = 1 - symmerty KL; 
      a = 1/2 - original Jenson-Shannon divergence.
    * If generalize:
        dist = dist + sum(P) - sum(Q).

    '''      
  
    x,y = _check_xy(x,y)
    bins,x,y = operators.ecdf(x,y)    
    dbins =  operators.diff(bins)
    x =x*dbins    
    y =y*dbins
 
    return kl(x,y, a, generalize) #np.sum(x*np.log(np.abs(x/(y+_EPS_))+_EPS_))
