import numpy as np
import scipy
_all__ = ['afft','cexp','cexp2pi','arg','polyval','gamma','corcof']

#-------------------------------------------------------------------
def afft(x, n_fft=None):
    '''
    Amplitude spectrum.
    ..math::
    afft = abs(fft(x))
    
    Parameters
    -----------
    * x: 1d ndarray.
    * n_fft: int or None,
        size of fft.
    
    Returns
    -----------
    * dx: 1d ndarary.    
    '''
    
    x = np.asarray(x)
    
    if n_fft is None:
        n_fft = x.size
    
    return np.abs(np.fft.fft(x))

#----------------------------
def arg(x):
    '''
    Unwraped phase.
    
    Parameters
    -----------
    * x: 1d ndarray (complex).
    
    Returns
    -----------
    * phase: 1d ndarary.    
    '''
    
    x = np.asarray(x)

    return np.unwrap(np.angle(x))
#--------------------------------------------
def corcof(a,b): 
    '''
    Correlation coefficient:
    ..math::
    corcof = sum(a*b^*)/sqrt(sum(a^2)sum(b^2))
    
     Parameters
    -----------
    * a,b: 1d ndarray (complex).
    
    Returns
    --------
    * corcof: float (complex).
    '''
    a = np.asarray(a)
    b = np.asarray(b)
    
    if a.shape!=b.shape:
        raise valueError('x.shape!=y.shape')
    if x.ndim >1:
        raise valueError('x.ndim >1')
        
    a = a - np.mean(a)
    b = b - np.mean(b)
    return np.sum(a*np.conj(b))/\
           np.sqrt(np.sum(np.square(a)
                          )*np.sum(np.square(b)))

#-------------------------------------------------------------------
cexp = lambda x: (np.exp(1j*x))
#-------------------------------------------------------------------
cexp2pi = lambda x, phi0=0: (np.exp(1j*(2*np.pi*x + phi0)))
  
#-------------------------------------------------------------------
def polyval(x,c):    
    '''
    Polynom of degree c.size-1 of input samples.
    
    Parameters
    -----------
    * x: input samples (1d ndarray).
    * c: polynom coefficients (1d nd array).
    
    Returns
    --------
     * polynom of degree c.shape-1.
    
    Notes
    -----------
    * Expression:
      ..math::        
      p(x) = c[0]+c[1]*x+c[2]*x^2+...+c[N-1]*x^(N-1),            
      where {c}_N are the polynom coefficients.

            
    '''    
    c = np.append([],np.asarray(c) )

    N = c.shape[0]
    
    x = np.asarray(x)    
    out = c[-1]    

    for i in np.arange(2, N + 1):
        out = c[N-i] + out*x   
    
    return out
#-------------------------------------------------------------------
def dif(y,delta=1):
    '''
    Difference of signal.
    
    Parameters
    -----------
    * x: 1d ndarray.
    * delta: float,
        step of dx.
    
    Returns
    --------
    * dx: 1d ndarray.
    
    '''
    ybar = (y[2:] - y[:-2])/2
    return np.concatenate(([y[1]-y[0]], ybar,[y[-1]-y[-2]] ))/delta

#--------------------------------------------------------------------  
def join_subsets(x, y):
    ''' 
    Join subsets (uinon of subsets). 
    
    Parameters:
    ------------
    * x,y: 1d ndarrays,
        input mainfolds.
    
    Returns:
    ---------
    * xy: 1d ndarray,
        the joint mainfolds. 
        
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    
    if x.shape!=y.shape:
        raise valueError('x.shape!=y.shape')
    if x.ndim >1:
        raise valueError('x.ndim >1')
        
    return np.max(np.vstack((x, y)),axis=0)
#     return np.asarray([np.max([hist_x[i],hist_y[i]]) for i in np.arange(hist_x.shape[0]) ])

#--------------------------------------------------------------------  
def cross_subsets(x, y):
    ''' 
    Cross subsets (intersection). 
    
    Parameters:
    ------------
    * x,y: 1d ndarrays,
        input mainfolds.
    
    Returns:
    ---------
    * xy: 1d ndarray,
        the cross of mainfolds. 
        
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    
    if x.shape!=y.shape:
        raise valueError('x.shape!=y.shape')
    if x.ndim >1:
        raise valueError('x.ndim >1')
        
    return np.min(np.vstack((x, y)),axis=0)
#     return np.asarray([np.min([hist_x[i],hist_y[i]]) for i in np.arange(hist_x.shape[0]) ])



#-------------------------------------------------------------------
__P_GAMMA__ = [ 676.5203681218851
               ,-1259.1392167224028
               ,771.32342877765313
               ,-176.61502916214059
               ,12.507343278686905
               ,-0.13857109526572012
               ,9.9843695780195716e-6
               ,1.5056327351493116e-7]

__EPSILON__ = 1e-07

def __drop_imag__(z):
    if np.abs(np.imag(z)) <= __EPSILON__:
        z = np.real(z)
    return z

def gamma(z):
    z = np.complex(z)
    if np.real(z) < 0.5:
        y = np.pi / (np.sin(np.pi*z) * gamma(1-z))  # Reflection formula
    else:
        z -= 1
        x = 0.99999999999980993
        for (i, pval) in enumerate(__P_GAMMA__):
            x += pval / (z+i+1)
        t = z + len(__P_GAMMA__) - 0.5
        y = np.sqrt(2*np.pi) * np.power(t,(z+0.5) )* np.exp(-t) * x
    return __drop_imag__(y)

