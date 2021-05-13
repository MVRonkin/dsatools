import numpy as np
import scipy

from ._lags_matrix import lags_matrix



from .. import utilits 

#  ktype: {exp, rbf, polynomial, sigmoid, linear, euclid, minkowsky, thin_plate, bump, polymorph}.
    
#     kpar: kernal parameter depends on the kernal type.
#----------------------------------------------------
__all__ = ['kernel_matrix','euclidian_matrix']

def kernel_matrix(x, mode = 'full', kernel='linear', 
                  kpar=1, lags = None, ret_base = False, normalization = True):
    '''
    
    Kernel matrix for input vector.
    Like an analogue of covariance_matrix
    
    Parameters
    -------------
    * x: input 1d ndarray.    
    * mode: string,
       mode = {'full',prew', 'postw', 'covar','traj', 'hanekl';'toeplitz'}   
       > mode = 'full': lags_matrix is the full toeplitz convolutional matrix 
           with dimentions [lags+N-1,lags],
            ..math::
            out = [ [x,0..0]^T,[0,x,0..0]^T,...,[0,..0,x]^T ],
            where: N is the x size.
       > mode =  'prew': lags_matrix is the prewindowed matrix with 
            first N columns of full matrix, and dimention = [N,lags];
       > mode = 'postw': lags_matrix is the postwindowed matrix with 
            last N columns of full matrix, and dimention = [N,lags];
       > mode = 'covar': lags_matrix is the trimmed full matrix with
            cut first and last m coluns (out = full[lags:N-lags,:]), 
            with dimention = [N-lags+1,lags];
       > mode = 'traj': lags_matrix is trajectory 
            or so-called caterpillar matrix 
            with dimention = [N,lags];
       > mode = 'hanekl':   M is the Hankel matrix 
            with dimention = [N,N];
       > mode = 'toeplitz': M is the symmetric Toeplitz matrix, 
            with dimention = [N,N].
    *  kernel: string,
        kernel = {exp, rbf, polynomial, sigmoid, linear, 
                euclid, minkowsky, thin_plate, bump, polymorph}.
    * kpar: kernel parameter, depends on the kernel type.            
    * lags: int or None,
        number of lags (N//2 dy default).
    * ret_base: bool,
        if true, than lags matrix will be also returned. 
    * normalization bool,
        if True, than matrix mean will be substructed.
        
    Returns
    -----------
    > ret_base is False:
        * kernel matrix: 2d ndarray.
    > ret_base is True:
        * matrix: 2d ndarray,
            kernel matrix.
        * lags_matrix: 
            lags matrix.
    Notes
    ---------
    * There are permissioned synonyms:
       'prew' and 'prewindowed';
       'postw' and 'postwindowed';
       'traj', 'caterpillar' and 'trajectory'.
    
    See also
    -----------
    covariance_matrix
    lags_matrix
    
    '''
    x = np.asarray(x)
    if (lags is None): lags = x.shape[0]//2
    
    base = lags_matrix(x, lags=lags, mode=mode)

    if(ret_base):
        out = base

    # TODO: if euclidian_matrix applied for base    
    if(kernel in ['rbf','thin_plate','euclid','minkowsky','bump',\
                  'polymorph','exp', 'laplacian','laplace','gauss']):
        base = base.T

    R = _kernel(base,base, ktype = kernel, kpar=kpar)

    if (normalization):
        column_sums = np.mean(R,axis=0) 
        total_sum   = np.mean(column_sums) 
        J = np.ones(R.shape[0])*column_sums
        R = R - J - J.T+total_sum 

    if(ret_base):
        out = (R,out) 
    else:
        out = R
    
    return out

#------------------------------------------------------------
def euclidian_matrix(X,Y, inner=False, square=True, normalize=False):
    '''
    Matrix of euclidian distance.
        I.E. Pairwise distance matrix.
    
    Parameters:
    ------------------
    * X,Y: 2d or 1d input ndarrays.
    * inner: bool,
        inner or outer dimesions.
    * square: bool,
        if false, than sqrt will be taken.
    * normalize: bool,
        if true, distance will be 
        normalized as d = d/(std(x)*std(y))    
        
    Returns
    ----------
    * out: 2d ndarray,
        pairwise distance matrix.    
    '''
    X,Y = _check_dim(X,Y)
    out = _euclid(X, Y, inner=inner)
    if not square: out = np.sqrt(out)
    if normalize: out /=np.std(X)*np.std(Y)   
        
    return out

#------------------------------------------------------------
def _kernel(a,b=None,ktype='rbf',kpar=1/2, take_mean = False):
    '''
    Kernel matrix (same as Gramm matrix).
    
    Parameters:
    ------------------
    * a,b: 2d or 1d input ndarrays.
    * ktype: string,
        ktype = {exp, rbf, polynomial, sigmoid, linear, 
            euclid, minkowsky, thin_plate, bump, polymorph}.
    * kpar: float,
        kernal parameter depends on the kernal type.

    '''
    a,b = _check_dim(a,b)

    k = np.zeros((a.shape[0],b.shape[0]), dtype = np.complex)
    
    if ktype is 'linear':   
        k = _linear(a,b)
    
    elif ktype is 'euclid': 
        k = _euclid(a,b) 
    
    elif ktype is 'minkowsky':
        k = np.power(_euclid(a,b), kpar/2)        
    
    elif ktype is 'sigmoid':
        k = np.tanh(_linear(a,b) + kpar)         
    
    elif ktype in ['rbf','gauss']:
        k = np.exp( - kpar * _euclid(a,b))  
    
    elif ktype in ['exp', 'laplacian','laplace']:
        k = np.exp( - kpar * np.sqrt(_euclid(a,b)))                
    
    elif ktype in ['poly','polynom','polynomial']:
        k = np.power(1+_linear(a,b), kpar) 
    
    elif ktype is 'thin_plate':
        k = _euclid(a,b)*np.log(_euclid(a,b))/2
        
    elif ktype is 'bump':
        k = _euclid(a,b)
        k = np.exp(-1/(1-kpar*k))
        
    elif ktype is 'polymorph':
        k  = np.power(_euclid(a,b), kpar/2)     
        k  = np.log(_euclid(a,b))/2
        k *= np.power(_euclid(a,b), (kpar-1)/2)
    
    elif ktype in ['rbf_inner']:
        k  = np.exp( - kpar * _euclid(a,b,True))     
    
    else:
        raise NotImplementedError('use one of the kernel from help')
    
    if (take_mean):
        column_sums = np.mean(R,axis=0) 
        total_sum   = np.mean(column_sums) 
        J = np.ones(R.shape[0])*column_sums
        R = R - J - J.T+total_sum 
    
    return k

#------------------------------------------------------------
def _check_dim(X,Y=None):
    X = np.asarray(X)
    
    if(Y is None):
        Y = X
    else:
        Y = np.asarray(Y)
        
    if X.ndim==1 and Y.ndim==1:
        X = utilits.to_2d(X,column=False)
        Y = utilits.to_2d(Y,column=False)
        
    elif X.ndim==1:
        X = utilits.to_2d(X,column=True)
    
    elif Y.ndim==1:
        Y = utilits.to_2d(Y,column=True)
    
#     elif X.ndim==2 and Y.ndim==2:
#         X = X.T
#         Y = Y.T

    
    return X,Y


#------------------------------------------------------------
def _linear(X, Y, open_dot=False):
    '''
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.linear_kernel.html
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/metrics/pairwise.py#L980
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/utils/extmath.py#L120
    '''

#     if a.ndim > 2 or b.ndim > 2:
    if(open_dot):
        ret = np.dot(X, np.conj(Y.T))
    else:
        ret = np.dot(X.T, np.conj(Y))
#     else:
#         ret = a @ b
    return ret

#------------------------------------------------------------
def _euclid(X, Y, inner= False):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/utils/extmath.py#L50
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/utils/extmath.py#L120
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/metrics/pairwise.py#L200
   
    """

    axis = 1
    open_dot = True
    
    if(inner):
        axis = 0
        open_dot = False        
    
    XX = np.sum(np.square(X),axis=axis)[:, np.newaxis]
    YY = np.sum(np.square(Y),axis=axis)[np.newaxis, :]
    
    distances = - 2 * _linear(X, Y,open_dot)
    
#     if(distances.shape[0] != XX.size):
#         distances = - 2 * _linear(X, Y,True)
    
    distances += XX

    distances += YY
    
    return np.abs(distances)



# def static_filter(p, g, n=None, squeeze=True):
#     """Compute the optimal cancellation filter from primary and secondary paths.

#     Note that this filter can be non-causal.

#     Parameters
#     ----------
#     p : array_like, shape (N[, L])
#         Primary path impulse response.
#     g : array_like, shape (N[, L[, M]])
#         Secondary path impulse response.
#     n : int
#         Output filter length.
#     squeeze: bool, optional
#         Squeeze output dimensions.

#     Returns
#     -------
#     numpy.ndarray, shape (n,[, M])
#         Optimal filter in frequency domain.

#     """
#     assert p.shape[0] == g.shape[0]

#     if n is None:
#         n = p.shape[0]

#     p = atleast_2d(p)
#     g = atleast_3d(g)

#     P = np.fft.fft(p, n=n, axis=0)
#     G = np.fft.fft(g, n=n, axis=0)

#     M = G.shape[2]

#     W = np.zeros((n, M), dtype=complex)
#     for i in range(n):
#         W[i] = - np.linalg.lstsq(G[i], P[i], rcond=None)[0]

#     return W if not squeeze else W.squeeze()


# def wiener_filter(x, d, n, g=None, constrained=False):
#     """Compute optimal wiener filter for single channel control.

#     From Elliot, Signal Processing for Optimal Control, Eq. 3.3.26

#     Parameters
#     ----------
#     x : array_like
#         Reference signal.
#     d : array_like
#         Disturbance signal.
#     n : int
#         Output filter length.
#     g : None or array_like, optional
#         Secondary path impulse response.
#     constrained : bool, optional
#         If True, constrain filter to be causal.

#     Returns
#     -------
#     numpy.ndarray, shape (n,)
#         Optimal wiener filter in freqency domain.

#     """
#     if g is None:
#         g = [1]

#     G = np.fft.fft(g, n=n)

#     # NOTE: one could time align the responses here first
#     _, Sxd = csd(x, d, nperseg=n, return_onesided=False)
#     _, Sxx = welch(x, nperseg=n, return_onesided=False)

#     if not constrained:
#         return - Sxd / Sxx / G

#     c = np.ones(n)
#     c[n // 2:] = 0
#     # half at DC and Nyquist
#     c[0] = 0.5
#     if n % 2 == 0:
#         c[n // 2] = 0.5

#     # minimum phase and allpass components of G
#     # NOTE: could also use https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.minimum_phase.html
#     Gmin = np.exp(np.fft.fft(c * np.fft.ifft(2 * np.log(np.abs(G)), n=n), n=n))
#     Gall = G / Gmin

#     # spectral factor
#     # NOTE: couuld also use https://github.com/RJTK/spectral_factorization/blob/master/spectral_factorization.py
#     F = np.exp(np.fft.fft(c * np.fft.ifft(np.log(Sxx), n=n), n=n))

#     h = np.ones(n)
#     h[n // 2:] = 0
#     return - np.fft.fft(h * np.fft.ifft(Sxd / F.conj() / Gall), n=n) / (F * Gmin)