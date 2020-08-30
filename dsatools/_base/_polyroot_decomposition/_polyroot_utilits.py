import numpy as np
import scipy

__all__ = ['roots2decomposition','roots2freqs']
#----------------------------------------
def roots2decomposition(x, roots, order=None, n_psd = None):
    '''
    Polynomial-roots method for signal decomposition.
    
    Parameters
    ------------
    * x: 1d ndarray, 
        signal for decomposed signal.
    * roots: 1d ndarray,
        roots of polynom to decomposition.
    * order : int or None,
        nuber of roots neares to the unit circle 
        in the Z-space ( roots.size if None).
    * n_psd: int or None,
        number of samples in decomposed signal
        x.size if None.

    Returns
    ------------
    * components: array with size [order, x.size].
    
    Notes
    -----------
    
    References
    -----------
    [1a] A. Fernandez-Rodriguez, L. de Santiago, 
    M.E.L. Guillen, et al.,"Coding Prony’s method 
    in MATLAB and applying it to biomedical signal filtering", 
        BMC Bioinformatics, 19, 451 (2018).
    [1b]  https://bmcbioinformatics.biomedcentral.com
        /articles/10.1186/s12859-018-2473-y   
    
    '''
    x = np.asarray(x)
    N = x.shape[0]
    roots = np.asarray(roots)
    
    if (n_psd is None): n_psd = N
        
    if order is None: order = roots.shape[0]

#         * stable_roots: bool,
#         if True, than only roots with module 
#         less or equal to 1 will be taken 
#         (i.e. dumped or stable sinusids).
#     if stable_roots:
#         roots = roots[np.abs(roots)<=1]

    roots = roots[np.imag(roots) != 0] # test on the correct roots

    #  extract roots numbers nearest to the circle with radius 1
    unicircle_idxs = np.argsort( np.abs(np.abs(roots)-1) )
    roots = roots[unicircle_idxs[:max(order,unicircle_idxs.size)]]
    
    # rest half of the values
    # i,m not sure thart it boost
    freqs = np.angle(roots)/2/np.pi
    roots = roots[freqs>0]
    # dumps = np.log(np.abs(roots)) 
    # roots = roots[np.abs(dumps)<=1]
    
    #TODO: replace on fft to increas stability
    v = np.vander(roots,N,increasing=True).T
    
    # TODO: look for fast way to solve this equation 
    # (based on vandermonde properties)
    h = scipy.linalg.lstsq(v,x)[0]   
    # amps  = np.abs(h); thets = np.angle(h)   
    out = np.zeros((min(order,roots.size),n_psd),dtype = x.dtype)   
    k = np.arange(n_psd)
    for i in np.arange(out.shape[0]):
        out[i,:] = h[i]*np.power(roots[i],k) 
    
    # finish details
    if out.shape[0]<order:
        rest = np.zeros(
          (order-out.shape[0],
           out.shape[1]),
          dtype = out.dtype)
        
        out = np.append(out, 
                        rest,
                        axis=0)
    
    return out

#----------------------------------------
def roots2freqs(roots, order=None, fs = 1):
    '''
    Polynomial-roots method for frequency 
        of near-unicircle signal components .
    
    Parameters
    ------------
    * roots: 1d ndarray,
        roots of polynom to decomposition.
    * order : int or None,
        nuber of roots neares to the unit circle 
        in the Z-space ( roots.size if None).
    * fs: float or None,
       sampling frequency.

    Returns
    ------------
    * freqs: array with size [order].
        
    Notes
    -----------
    * for ar2freq!!
    
    
    References
    -----------
    [1a] A. Fernandez-Rodriguez, L. de Santiago, 
    M.E.L. Guillen, et al.,"Coding Prony’s method 
    in MATLAB and applying it to biomedical signal filtering", 
        BMC Bioinformatics, 19, 451 (2018).
    [1b]  https://bmcbioinformatics.biomedcentral.com
        /articles/10.1186/s12859-018-2473-y   
    
    '''
    roots = np.asarray(roots)
        
    if order is None: order = roots.shape[0]

    roots = roots[np.imag(roots) != 0] # test on the correct roots

    #  extract roots numbers nearest to the circle with radius 1
    unicircle_idxs = np.argsort( np.abs(np.abs(roots)-1) )
    roots = roots[unicircle_idxs[:max(order,unicircle_idxs.size)]]
    
    # rest half of the values
    # i,m not sure thart it boost
    freqs = fs*np.angle(roots)/2/np.pi    
#     dumps = fs*np.log(np.abs(roots[freqs>0])) 
    freqs = freqs[freqs>0]
    
    # finish details
    if freqs.shape[0]<order:
        rest = np.zeros((order-freqs.shape[0]))
        freqs = np.append(freqs,rest)
#         dumps = np.append(dumps,rest)
    
    return freqs[:order]
