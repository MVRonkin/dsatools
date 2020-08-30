import numpy as np
import scipy  

from ... import operators

__all__ = ['subspace2psd','subspace2freq']


#------------------------------------
def subspace2psd(subspace, n_psd = None ):
    '''  
    Subspace to pseudospetrum (psd) transformation,
        uisng maximum entropy principle. 

    Parameters
    --------------
    * subspace: 2d ndarray,
        input subspace with 
        dimentions = [lags,space_size].
    * n_psd: int or None,
        number of points in estimated psd, 
        (subspace.shape[0]*2 by default).
    
    Returns
    -------------
    * psd: 1d ndarray,
        pseudospectrum of input subspace.
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled.
    
    Example
    -------------
        signal1 = signals.generator.harmonics(amplitude=[1],
                        f0=[1,2,3,3.1,2.04],
                        delta_f=[0.4],fs=10)
        ut.probe(signal1)
        order = 15
        x = signal1
        noise_space = music(signal1, order, mode='full', lags=None)
        psd = subspace2psd(noise_space )
        plt.plot(psd); plt.show()
     
    References
    ----------------
    [1a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page
    [2a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling
    
    '''
    
    if n_psd is None: n_psd = 2*subspace.shape[0]

    psd = 0

    for i in range(subspace.shape[1]):
        psd += np.abs( np.fft.ifft(subspace[:,i],n_psd, axis=0))
    
    return 1/psd



#------------------------------------
def subspace2freq(subspace,order=None,fs=1):
    '''  
    Subspace to pseudospetrum (psd) transformation,
        using alternative method. 

    Parameters
    --------------
    * subspace: 2d ndarray,
        input subspace with 
        dimentions = [lags,space_size].
    * order: int or None,
        order of the model,
        subspace.shape[1] - subspace.shape[0],
        for default (if noise subspace is used).
    fs: float,
        sampling frequency.
    
    Returns
    -------------
    * psd: 1d ndarray,
        pseudospectrum of input subspace.
   
    Notes
    -----------


    Example
    -------------
        signal0 = signals.generator.harmonics(amplitude=[1],
                            f0=[1,2,3],
                            delta_f=[0.0],fs=10)

        ut.probe(signal0)

        order = 3
        x = signal0
        noise_space = music(signal0, order, mode='full', lags=None)
        freqs = subspace2freq(noise_space,order )
        print(freqs)
   
    References
    ----------------
    [1a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page
    [2a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling
        
    '''
    if fs is None: fs = 1
    
    subspace = np.matrix(subspace)
    
    if order is None:
        order = subspace.shape[0]-subspace.shape[1]
        
    P=subspace*subspace.H
    N = subspace.shape[0]*2

    Q=np.zeros(N,dtype=P.dtype)
    for idx,val in enumerate(np.arange(N//2-1,-N//2,-1)):
        Q[idx]=np.sum(np.diag(P,val))

    roots = np.roots(Q)

    roots = roots[np.abs(roots)<1]
    roots = roots[np.imag(roots) != 0]
 
    # extract roots numbers nearest to the circle with radius 1
    unicircle_idxs = np.argsort( np.abs(np.abs(roots)-1) )
    roots = roots[unicircle_idxs[:order]]

    f= -fs*np.angle(roots)/(2.*np.pi)

    return f[f>0]

