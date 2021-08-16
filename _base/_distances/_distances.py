import numpy as np

_EPS_ = 1e-7
# ----------------------------------------------------------------------------------------


def _check_xy(x, y=None):
    x = np.asarray(x)
    if y is None:
        y = x
    else:
        y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError('x.shape != y.shape')

    return x, y
# ----------------------------------------------------------------------------------------


def _entropy(x, y):
    '''
    Entropy of two signals.

    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray.

    Returns
    --------
    * entropy: float (or complex). 

    Notes
    ---------
    * H(x|y) = sum(x*ln(|y|)). 

    '''
    return np.sum(x*np.log(np.abs(y)+_EPS_))
# ----------------------------------------------------------------------------------------


def _kl(x, y):
    '''
    Kullback-Leibler (KL) divergence of two signals.

    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray.

    Returns
    --------
    * divergence: float (or complex). 

    Notes
    ---------
    * kl(x||y) = sum(x*ln(|x/y|)).

    '''
    return np.sum(x*np.log(np.abs(x/(y+_EPS_))+_EPS_))
# ----------------------------------------------------------------------------------------


def _log(x):
    '''
    logarithm of module.

    Parameters
    ----------
    * x: 1d ndarray.

    Returns
    --------
    * log: 1d ndarray. 

    '''
    return np.log(np.abs(x)+_EPS_)
# ----------------------------------------------------------------------------------------


def euclidian(x, y, square=False):
    '''
    Euclidian distance.
    ..math::
       Distance = sum(|[x-y]^2|)^(1/2)

    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray.
    * square: bool,
        if True, than square of 
        distance value will be returned.

    Returns
    --------
    * distance: float (or complex).  

    Notes
    ------
    * If square: Distance = sum([x-y]^2),
        else:    Distance = sqrt(sum([x-y]^2)).

    '''
    x, y = _check_xy(x, y)
    out = np.sum(np.abs(np.square(x - y)))
    if not square:
        out = np.sqrt(out)
    return out
# ----------------------------------------------------------------------------------------


def minkowsky(x, y, p=2, root=False):
    '''
    minkowskydistance.
    ..math::
      Distance = sum(|[x-y]^p|)^(1/p)

    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray.
    * p: float,
        degree.
    * root: bool,
        if True, than root of 
        distance value will be returned.

    Returns
    --------
    * distance: float (or complex).  

    Notes
    ------
    * If root: Distance = sum(|[x-y]^p|)^(1/p),
        else:  Distance = (sum(|[x-y]^p|)).

    '''
    x, y = _check_xy(x, y)

    #out = np.sum(np.abs(np.power(x - y, p)))
    out = np.sum(np.power(np.abs(x-y), p))
    if root and p != 0:
        out = np.power(out, 1/p)

    return out
# ----------------------------------------------------------------------------------------


def correlation(x, y, normalize=False):
    '''
    Correlation coefficients of two signals.
    ..math::	
        corcof = sum(xy^*)/(sum(x^2)sum(y^2))^0.5

    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray.
    * normalize: bool,
        if True, than normalized correlation
        coefficients will be returned.

    Returns
    --------
    * cof: float (or complex).  

    Notes
    ------
    * If normalize: cof = sum(x*conj(y)),
        else: cof = sum(x*conj(y))/sqrt(sum(x*conj(x))*sum(y*conj(y))).
    '''
    x, y = _check_xy(x, y)
    out = np.sum(x*np.conj(y))

    if (normalize):
        out /= np.sqrt(np.abs(np.sum(np.square(x))*np.sum(np.square(y))))

    return out
# ----------------------------------------------------------------------------------------


def angle(x, y, normalize=False):
    '''
    Angle between of two signals.
    ..math::	
        angle  = arctan2[Im{corcof},Re{corcof}] if x complex valued 
        angle  = arccos[corcof] if x real valued 

    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray.
    * normalize: bool,
        if True, than normalized values
        will be taken into angle.

    Returns
    --------
    * angle: float.  

    Notes
    ------
    * If real values: angle = arccos(correlation_cof)
        else: angle = arctan2( Im{correlation_cof}, Re{correlation_cof})
    * If normalize: correlation_cof = sum(x*conj(y)),
        else: correlation_cof = sum(x*conj(y))/sqrt(sum(x*conj(x))*sum(y*conj(y))).
    * Normalization is necessary if inputs are real valued signals.

    '''

    x, y = _check_xy(x, y)
    out = np.sum(x*np.conj(y))
    if (normalize):
        out /= np.sqrt(np.abs(np.sum(np.square(x))*np.sum(np.square(y))))

    # if any iscomplex
    if x.dtype in [np.complex, complex, np.complex128, np.complex64] or \
            y.dtype in [np.complex, complex, np.complex128, np.complex64]:
        out = np.angle(out)

    else:
        out /= np.sqrt(np.abs(np.sum(np.square(x))*np.sum(np.square(y))))
        out = np.arccos(out)

    return out
# ----------------------------------------------------------------------------------------


def entropy(x, y=None, symmetry_entropy=False, cross_information=False):
    '''
    Entropy of two signals H(x|y).
    ..math::
       H(x|y) = sum(x*ln(|y|))

    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray,
        if None, y = x.
    * symmetry_entropy: bool,
        if True, than the symmetry entropy 
        (H(x|y)+H(y|x)) will be returned.
    * cross_information: bool,
        if True, than cross entropy 
        (H(x|y)-H(x|x)-H(y|y)) or
        (H(x|y)+H(y|x)-H(x|x)-H(y|y) for symmetry case) 
        will be returned.    

    Returns
    --------
    * entropy: float (or complex). 

    Notes
    ---------
    * Entropy calculaed as
        H(x|y) = sum(x*ln(|y|))
    '''
    x, y = _check_xy(x, y)

    out = _entropy(x, y)

    if(symmetry_entropy):
        out += _entropy(y, x)

    if(cross_information):
        out += -_entropy(x, x) - _entropy(y, y)

    return out

# ----------------------------------------------------------------------------------------


def itakura_saito(x, y=None, p=2, spectra_domain=True):
    '''
    Itakura Saito distance of two signals.
    ..math::    
     I(x|y) = x/y - log(|x|)/log(|y|)-1

    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray,
        if None, y = x.
    * p: float,
        degree.
    * spectra_domain: bool,
        if True, than distance in spectrum
        domain will be returned.

    Returns
    --------
    * distance: float (or complex). 

    Notes
    ---------
    * Itakura Saito distance calculaed as
        I(x|y) = x/y - log(|x|)/log(|y|)-1

    '''

    if(spectra_domain):
        x = np.fft.fft(x)
        y = np.fft.fft(y)

    out = x/(y + _EPS_) - _log(x)/(_log(y)+_EPS_)-1

    if p != 1:
        out = np.power(np.abs(out), p)
    out = np.sum(out)

    if p != 0:
        out = np.power(np.abs(out), 1/p)

    return np.abs(out)

# ----------------------------------------------------------------------------------------


def kl(x, y, a=None, generalize=False):
    ''' 
    Kullback-Leibner (KL) divergence of
    two signals kl(x|y).
    ..math::       
        dist = KL(x||a*y+(1-a)*x)+KL(y||a*x+(1-a)*x)

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
        dist = kl(x||y) = sum(x*ln(|x/y|))
    * If (a>=0) it will be alpha-Jenson-Shannon Divergence: 
        dist = KL(x||a*y+(1-a)*x)+KL(y||a*x+(1-a)*x)
      a = 1 - symmerty KL; 
      a = 1/2 - original Jenson-Shannon divergence.
    * If generalize:
        dist = dist + sum(x) - sum(y).

    '''
    x, y = _check_xy(x, y)
    if a is None:
        a = -1

    # Jensen-Shannon divergence
    if a > 0:
        out = _kl(x, y*a+x*(1-a)) + _kl(y, x*a+y*(1-a))

        if(generalize):
            out += np.sum(x)-np.sum(y)

    else:  # a=<0
        out = _kl(x, y)

        if generalize:
            out += np.sum(x)

    return out

# ----------------------------------------------------------------------------------------


def dice(x, y):
    ''' 
    Dice coefficient of two signals.
    ..math::
         dice = 2KL(x||y)/(H(x|y)+H(y|x))

    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray,

    Returns
    --------
    * dice: float (or complex). 

    Notes
    ---------
    * Dice divergence calculaed as
        dice = 2KL(x||y)/(H(x|y)+H(y|x)),
        * KL(x||y) = sum(x*ln(|x/y|)) - equal to mutual information.
        * H(x|y) = x*ln(|y|) - entroy.

    '''
    return 2*_kl(x, y)/(_entropy(x, x)+_entropy(y, y))

# ----------------------------------------------------------------------------------------


def jaccard(x, y):
    ''' 
    Jaccard coefficient of two signals.
    ..math::
        jaccard = 2KL(x||y)/max(H(x|y),H(y|x))

    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray.

    Returns
    --------
    * jaccard: float (or complex). 

    Notes
    ---------
    * Jaccard divergence calculaed as
        jaccard = 2KL(x||y)/max(H(x|y),H(y|x)),
        * KL(x||y) = sum(x*ln(|x/y|)) - equal to mutual information.
        * H(x|y) = x*ln(|y|) - entroy.

    '''
    return 2*_kl(x, y)/max(_entropy(x, x), _entropy(y, y))

# ----------------------------------------------------------------------------------------


def selfentropy_reiny(x, a=1):
    ''' 
    Alpha Divergence based on the Reiny Entropy.

    Parameters
    ----------
    * x: 1d ndarray.
    * a: float,
        entropy parameter.

    Returns
    --------
    * entropy: float (or complex).

    Notes
    ---------
    * Self entropy, depends on a:
        H(x_alpha)=log(sum(x^a))/(1-a)

        if(a=0): H(x_alpha) = len(x)        
        if(a=1): H(x_alpha) = sum(x*log(x))

        if(a='max'): H(x_alpha) =log(max(x))
        if(a='min'): H(x_alpha) =log(min(x))

    See Also
    ----------------
    * alpha_divergence
    * entropy
    '''
    x = np.asarray(x)

    if(a == 0):
        return np.log(x.size)
    elif(a == 1):
        return _entropy(x, x)
    elif(a == 'max'):
        return _log(np.max(x))
    elif(a == 'min'):
        return _log(np.min(x))
    else:
        return _log(np.sum(np.power(x, a)))

# ----------------------------------------------------------------------------------------


def alpha_divergence(x, y, a=1):
    ''' 
    Alpha Divergence based on the Reiny Entropy

    Parameters
    ----------
    * x: 1d ndarray.
    * y: 1d ndarray.
    * a: float,
        divergence parameter.

    Returns
    --------
    * divergence: float (or complex).

    Notes
    -------------
    Alpha Divergence expressions depends on a:
        D(x||y) = log(sum(p^a*q^(1-a)))/(a-1)

        if(a==0):   D = -log(sum(q*sign(p)));        
        if(a==0.5): D = -2log(sum(sqrt(p*q)));        
        if(a==1):   D = sum(p*log(p/q));
        ...
        if(a=='max'): D = log(max(p/q));        
        if(a=='min'): D = log(min(p/q));        
        if(a=='sum'): D = log(sum(p/q)).


    '''
    if(y is None):
        return entropy_reiny(x, a=a)

    x, y = _check_xy(x, y)

    if(a == 0):
        return _log(np.sum(y*np.sign(x)))

    elif(a == 0.5):
        return -2*_log(np.sum(np.sqrt(np.abs(x*y))))

    elif(a == 1):
        return _kl(x, y)

    elif(a == 'max'):
        return _log(np.max(x/y))

    elif(a == 'min'):
        return _log(np.min(x/y))

    elif(a == 'sum'):
        return _log(np.sum(x/y))

    else:
        p = np.power(x, a)
        q = np.power(y, 1-a)
        return _log(np.sum(p*q))/(a-1)

# ----------------------------------------------------------------------------------------


def ab_divergence(x, y, a=1, b=1):
    ''' 
        alpha-betta Divergence
        https://arxiv.org/pdf/1805.01045.pdf 
        if(a==0  and b==0): AB = sum((log(x)-log(y))^2)/2
        if(a==0  and b!=0): AB = sum(x^b)/sum(y^b) - b*sum((x^b)*log(x/y))/b^2
        if(a!=0  and b==0): AB = sum(y^a)/sum(x^a) - a*sum((y^a)*log(y/x))/a^2
        if(a==-b and b!=0): AB = (log(sum(x^a/y^a)) - sum(log(x^a/y^a)))/a^2
        else(a+b!=0, b!=0): AB = log[sum(x^(a+b))^(a/(a+b)) * sum(y^(a+b))^(b/(a+b))/sum(x^a * y^b)]/(a*b)
    '''
    x, y = _check_xy(x, y)

    if(a == 0 and b == 0):
        return np.sum(np.square(_log(x/y)))/2

    elif(a == 0 and b != 0):
        if(b == 1):
            return np.sum(x)/np.sum(y) - np.sum(x*_log(x/y))
        p = np.power(x, b)
        q = np.power(y, b)
        return (np.sum(p)/np.sum(q) - b*np.sum(p*_log(x/y)))/np.square(b)

    elif(a != 0 and b == 0):
        if(a == 1):
            return np.sum(y)/np.sum(x) - np.sum(y*_log(y/x))
        p = np.power(x, a)
        q = np.power(y, a)
        return (np.sum(q)/np.sum(p) - a*np.sum(q*_log(y/x)))/np.square(a)

    elif(a == -b and a != 0):
        if(a == 1):
            return np.log(np.sum(x/y))-np.sum(_log(x/y))
        p = np.power(x, a)
        q = np.power(y, a)
        return (np.log(np.sum(p/q))-np.sum(_log(p/q)))/np.square(a)
    else:
        p = np.power(x, a+b)
        q = np.power(y, b+a)
        out1 = np.power(np.sum(p), a/(a+b))*np.power(np.sum(q), b/(a+b))

        p = np.power(x, a)
        q = np.power(y, b)
        out2 = np.sum(p*q)

        return _log(out1/out2)*(1/a/b)


# #----------------------------------------------------------------------------------------
# def entropy(x,y=None, symmetry_entropy = False, cross_information = False):

#     x,y = _check_xy(x,y)

#     x = np.asarray(x,dtype = np.complex)
#     y = np.asarray(y,dtype = np.complex)

#     out = x*np.log(y)

#     if(symmetry_entropy):
#         out +=  y*np.log(x)

#     if(cross_information):
#         out += -x*np.log(x) - y*np.log(y)

#     return np.abs(np.nansum(out))

# #--------------------------------------------------------
# def mutualDistribution(p,q=None, degree=__DEGREE__, alpha = __ALPHA__, distr_mode = None, auxilay_value = 1):
#         '''
#             modes: '','None','Minkovsky','corCof','naive', 'normal','polynom',
#             'mutualMax','mutualMin','mutualMinMax'

#             auxilay_value is std in normal mode

#             alpha need for: q = p*(1-alpha)+q*alpha
#         '''

#         if (distr_mode not in __MUTUAL_DISTR_MODES__):
#             raise ValueError('distr_mode does not corresponds to \
#                              the mutual_distr_modes, use get_mutual_distr_modes() for get lsit of modes ')

#         p = np.asarray(p)+__EPSILON__
#         if(q is None):
#             q = p
#         else:
#             q = np.asarray(q)+__EPSILON__

#         q = p*(1-alpha)+q*alpha

#         out = q

#         if(distr_mode in ['None',None,'']):
#             out = __power__(q,degree)

#         elif(distr_mode == 'Minkovsky'):
#             out =  __power__((p-q),degree)

#         elif(distr_mode == 'corCof'):
#             out =  __power__((p*q),degree)

#         elif(distr_mode == 'naive'):
#             out = __power__(q,degree)

#         elif(distr_mode == 'normal'):
#             pp = __power__(p, degree)
#             qq = __power__(q, degree)
#             out = np.exp( ((pp-qq),2) /np.power(auxilay_value,2) )

#         elif(distr_mode == 'polynom'):
#             out = __power__(1-p*q, degree)

#         elif(distr_mode == 'mutualMax'):
#             pp = __power__(p, degree)
#             qq = __power__(q, degree)
#             out = [max(pi,qi) for pi,qi in zip(pp,qq)]
#             out = np.asarray(out)

#         elif(distr_mode == 'mutualMin'):
#             pp = __power__(p, degree)
#             qq = __power__(q, degree)
#             out = [min(pi,qi) for pi,qi in zip(pp,qq)]
#             out = np.asarray(out)

#         elif(distr_mode == 'mutualMinMax'):
#             pp = __power__(p, degree)
#             qq = __power__(q, degree)
#             out = [min(pi,qi)/max(pi,qi) for pi,qi in zip(pp,qq)]
#             out = np.asarray(out)

#         return out
# #--------------------------------------------------------
# def get_mutual_distribution_modes():
#     return __MUTUAL_DISTR_MODES__

# #--------------------------------------------------------

# #---------------------------
# def __power__(x,degree):
#     out = 0
#     if(degree == 0):
#         out = np.log(np.abs(x))
#     elif(degree == 1):
#         out = x
#     elif(degree == 0.5):
#         out = np.sqrt(x)
#     else:
#         out = np.power(x,degree)
#     return out
