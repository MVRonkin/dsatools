import numpy as np
import scipy  

from ... import operators

__all__ = ['music',
           'ev',
           'pisarenko',
           'pisarenko_cor',
           'minvariance',
           'kernel_noisespace',
           'kernel_signalspace']

#----------------------------------------------
def music(x, order, mode='full', lags=None):    
    '''  
    Estimation the MUltiple SIgnal Classification 
      (MUSIC) algorithm model, determinated as noise 
      subspace signal part.

    Parameters
    --------------
    * x: 1d ndarray,
        input signal.
    * order: int,
        order of the model
        (number of valuable components, 
        size of signal subspace). 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    * lags: int or None,
        number of lags in correlation function, 
        (x.shape[0]//2 by default).
    
    Returns
    -------------
    * noise_subspace: 2d ndarray,
        estimated noise subspace with 
        dimentions = [lags,lags-order].
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * Use roots in the function:
        subspace2psd;
        subspace2freqs.
        
    Example
    -------------
        signal1 = signals.generator.harmonics(amplitude=[1],
                        f0=[1,2,3,3.1,2.04],
                        delta_f=[0.4],fs=10)
        signal0 = signals.generator.harmonics(amplitude=[1],
                            f0=[1,2,3],
                            delta_f=[0.0],fs=10)

        ut.probe(signal1)
        ut.probe(signal0)

        order = 15
        x = signal1
        noise_space = music(signal1, order, mode='full', lags=None)
        psd = subspace2psd(noise_space )
        plt.plot(psd); plt.show()

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
    x = np.asarray(x)
    N = x.shape[0]       

    if(lags is None): lags = N//2

    R = operators.covariance_matrix(x, lags=lags, mode=mode)
    
       
    #TODO: in my practice eigen value always sorted
    # from the highest to the lowest, but probably sorting is required
    es,ev=np.linalg.eig(R)   
#     idxs = es.argsort()[::-1]
#     ev = ev[:,idxs]
    
    noise_space = np.matrix(ev[:,order:])
    
    return noise_space

#-----------------------------------------
def ev(x, order, mode='full', lags=None):    
    '''  
    Estimation the Eigen Values (EV) 
      algorithm model, determinated as 
      normalized noise subspace signal part.

    Parameters
    --------------
    * x: 1d ndarray,
        input signal.
    * order: int,
        order of the model
        (number of valuable components, 
        size of signal subspace). 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    * lags: int or None,
        number of lags in correlation function, 
        (x.shape[0]//2 by default).
    
    Returns
    -------------
    * noise_subspace: 2d ndarray,
        estimated noise subspace with 
        dimentions = [lags,lags-order].
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * Use roots in the function:
        subspace2psd;
        subspace2freqs.
    
    Example
    -------------
        signal1 = signals.generator.harmonics(amplitude=[1],
                    f0=[1,2,3,3.1,2.04],
                    delta_f=[0.4],fs=10)
        signal0 = signals.generator.harmonics(amplitude=[1],
                            f0=[1,2,3],
                            delta_f=[0.0],fs=10)
        ut.probe(signal1)
        ut.probe(signal0)

        order = 15
        x = signal1
        noise_space = ev(x, order, mode='full', lags=None)
        psd = subspace2psd(noise_space )
        plt.plot(psd); plt.show()

        order = 3
        x = signal0
        noise_space = ev(x, order, mode='full', lags=None)
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
    x = np.asarray(x)
    N = x.shape[0]       

    if(lags is None): lags = N//2

    R = operators.covariance_matrix(x, lags=lags, mode=mode)
    
       
    #TODO: in my practice eigen value always sorted
    # from the highest to the lowest, but probably sorting is required
    es,ev=np.linalg.eig(R)   
#     idxs = es.argsort()[::-1]
#     ev = ev[:,idxs]
    
    noise_space = np.matrix(ev[:,order:]/es[order:])
    
    return noise_space

#------------------------------------
def pisarenko(x, order, mode='full'):    
    '''  
    Estimation the Pisarenko harmonic decomposition
        algorithm, determinated as noise subspace 
        signal part, using first noise subspace vector.
        
    Parameters
    --------------
    * x: 1d ndarray,
        input signal.
    * order: int,
        order of the model
        (number of valuable components, size of signal subspace),
        and the number lags of covariance matrix - 1. 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    
    Returns
    -------------
    * noise_subspace: 2d ndarray,
        estimated noise subspace with 
        dimentions = [lags,lags-order].
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * Use roots in the function:
        subspace2psd;
        subspace2freqs.
    
    Example
    -------------
        signal1 = signals.generator.harmonics(amplitude=[1],
                        f0=[1,2,3,3.1,2.04],
                        delta_f=[0.4],fs=10)
        signal0 = signals.generator.harmonics(amplitude=[1],
                            f0=[1,2,3],
                            delta_f=[0.0],fs=10)

        ut.probe(signal1)
        ut.probe(signal0)

        order = 190
        x = signal1
        noise_space = pisarenko(x, order, mode='full')
        psd = subspace2psd(noise_space )
        plt.plot(psd); plt.show()

        order = 3
        x = signal0
        noise_space = pisarenko(x, order, mode='full')
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
    x = np.asarray(x)
    N = x.shape[0]       

    R = operators.covariance_matrix(x, lags=order+1, mode=mode)
    
       
    #TODO: in my practice eigen value always sorted
    # from the highest to the lowest, but probably sorting is required
    es,ev=np.linalg.eig(R)   
#     idxs = es.argsort()[::-1]
#     ev = ev[:,idxs]
    
    noise_space = np.matrix(ev[:,order:]/es[order:])
    
    return noise_space


#------------------------------------
def pisarenko_cor(x, order, mode='full', cor_mode = 'straight'):    
    '''  
    Estimation the Pisarenko harmonci decomposition
        algorithm, determinated as noise subspace 
        signal part, using first noise subspace vector
        of the additionally obtained 
        signal correlation function.
        
    Parameters
    --------------
    * x: 1d ndarray,
        input signal.
    * order: int,
        order of the model
        (number of valuable components, size of signal subspace),
        and the number lags of covariance matrix - 1. 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    * cor_mode: string,
        mode of additionally taken correlation function,
        cor_mode = {full,same,straight}.

    Returns
    -------------
    * noise_subspace: 2d ndarray,
        estimated noise subspace with 
        dimentions = [lags,lags-order].
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * Use roots in the function:
        subspace2psd;
        subspace2freqs.
    
    Example
    -------------
       signal1 = signals.generator.harmonics(amplitude=[1],
                        f0=[1,2,3,3.1,2.04],
                        delta_f=[0.4],fs=10)
        signal0 = signals.generator.harmonics(amplitude=[1],
                            f0=[1,2,3],
                            delta_f=[0.0],fs=10)

        ut.probe(signal1)
        ut.probe(signal0)

        order = 190
        x = signal1
        noise_space = pisarenko_cor(x, order, mode='full')
        psd = subspace2psd(noise_space )
        plt.plot(psd); plt.show()

        order = 3
        x = signal0
        noise_space = pisarenko_cor(x, order, mode='covar')
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
    x = np.asarray(x)
    N = x.shape[0]       
    r = operators.correlation(x,mode = cor_mode)
    R = operators.covariance_matrix(r, lags=order+1, mode=mode)
    
       
    #TODO: in my practice eigen value always sorted
    # from the highest to the lowest, but probably sorting is required
    es,ev=np.linalg.eig(R)   
#     idxs = es.argsort()[::-1]
#     ev = ev[:,idxs]
    
    noise_space = np.matrix(ev[:,order:]/es[order:])
    
    return noise_space

#------------------------------------
def minvariance(x, order, mode='full'):    
    '''  
    Estimation of the pseudo-spectrum based on the 
        minimum variance (maximum likelyhood) algorithm,
        using normalized signal subspace. 

    Parameters
    --------------
    * x: 1d ndarray,
        input signal.
    * order: int,
        order of the model
        (number of valuable components, size of signal subspace),
        and the number lags of covariance matrix . 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    
    Returns
    -------------
    * noise_subspace: 2d ndarray,
        estimated noise subspace with 
        dimentions = [lags,lags-order].
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * Use roots in the function:
        subspace2psd;
        subspace2freqs.
    
    Example
    -------------
        signal1 = signals.generator.harmonics(amplitude=[1],
                    f0=[1,2,3,3.1,2.04],
                    delta_f=[0.4],fs=10)
        signal0 = signals.generator.harmonics(amplitude=[1],
                            f0=[1,2,3],
                            delta_f=[0.0],fs=10)

        ut.probe(signal1)
        ut.probe(signal0)

        order = 190
        x = signal1
        noise_space = minvariance(x, order, mode='full')
        psd = subspace2psd(noise_space )
        plt.plot(psd); plt.show()

        order = 3
        x = signal0
        noise_space = minvariance(x, order, mode='full')
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
    x = np.asarray(x)
    N = x.shape[0]       

    R = operators.covariance_matrix(x, lags=order, mode=mode)
    
       
    #TODO: in my practice eigen value always sorted
    # from the highest to the lowest, but probably sorting is required
    es,ev=np.linalg.eig(R)   
#     idxs = es.argsort()[::-1]
#     ev = ev[:,idxs]
    
    space = np.matrix(ev[:,:]/es[:])
    
    return space

#------------------------------------
def kernel_noisespace(x, order, mode='full', 
                 kernel = 'linear', kpar=1, lags=None, use_ev = False):    
    '''  
    Estimation the noise subspace based  
        model of the kernel based signal.

    Parameters
    --------------
    * x: 1d ndarray,
        input signal.
    * order: int,
        order of the model
        (number of valuable components, 
        size of signal subspace). 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    * kernel: string,
        kernel trick function,
        kernel type = {linear, poly, rbf, thin_plate}.
    * kpar: float,
        kernel parameter, depends on the kernel type.
    * lags: int or None,
        number of lags in correlation function, 
        (x.shape[0]//2 by default).
    * use_ev: bool,
        if True, than normalized space will be taken.
    
    Returns
    -------------
    * noise_subspace: 2d ndarray,
        estimated noise subspace with 
        dimentions = [lags,lags-order].
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * Use roots in the function:
        subspace2psd;
        subspace2freqs.
    
    Example
    -------------
        signal1 = signals.generator.harmonics(amplitude=[1],
                    f0=[1,2,3,3.1,2.04],
                    delta_f=[0.4],fs=10)
        signal0 = signals.generator.harmonics(amplitude=[1],
                            f0=[1,2,3],
                            delta_f=[0.0],fs=10)

        ut.probe(signal1)
        ut.probe(signal0)

        order = 190
        x = signal1
        noise_space = kernel_noisespace(x, order, mode='full', kernel='rbf',kpar=0.0001, use_ev=False)
        psd = subspace2psd(noise_space )
        plt.plot(psd); plt.show()

        order = 3
        x = signal0
        noise_space = kernel_noisespace(x, order, mode='full', kernel='rbf',kpar=0.0001, use_ev=False)
        freqs = subspace2freq(noise_space,order )
        print(freqs)
        
    References
    ----------------
    [1a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page.
    [2a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling.
    
    '''     
    x = np.asarray(x)
    N = x.shape[0]       

    if(lags is None): lags = N//2

    R = operators.kernel_matrix(x,
                                mode   = mode,
                                kernel = kernel,
                                kpar   = kpar,
                                lags   = lags,
                                ret_base=False,
                                normalization=True)
    
       
    #TODO: in my practice eigen value always sorted
    # from the highest to the lowest, but probably sorting is required
    es,ev=np.linalg.eig(R)   
#     idxs = es.argsort()[::-1]
#     ev = ev[:,idxs]
    
    if use_ev:
        noise_space = np.matrix(ev[:,order:]/es[order:])
    else:
        noise_space = np.matrix(ev[:,order:])

    return noise_space

#------------------------------------
def kernel_signalspace(x, order, mode='full', 
                 kernel = 'linear', kpar=1, lags=None, use_ev = False):    
    '''  
    Estimation the signal subspace based  
        model of the kernel based signal.

    Parameters
    --------------
    * x: 1d ndarray,
        input signal.
    * order: int,
        order of the model
        (number of valuable components, 
        size of signal subspace). 
    * mode: string,
        mode of least-square problem solution,
        mode = {full,toeplitz,covar,traj,prew,postw}.
    * kernel: string,
        kernel trick function,
        kernel type = {linear, poly, rbf, thin_plate}.
    * kpar: float,
        kernel parameter, depends on the kernel type.
    * lags: int or None,
        number of lags in correlation function, 
        (x.shape[0]//2 by default).
    * use_ev: bool,
        if True, than normalized space will be taken.
    
    Returns
    -------------
    * noise_subspace: 2d ndarray,
        estimated noise subspace with 
        dimentions = [lags,lags-order].
   
    Notes
    -----------
    * If signal is real-valued, 
        than order of the model shold be douled with
        respect to complex case.
    * Use roots in the function:
        subspace2psd;
        subspace2freqs.
    
    Example
    -------------
        signal1 = signals.generator.harmonics(amplitude=[1],
                    f0=[1,2,3,3.1,2.04],
                            delta_f=[0.4],fs=10)
        signal0 = signals.generator.harmonics(amplitude=[1],
                            f0=[1,2,3],
                            delta_f=[0.0],fs=10)

        ut.probe(signal1)
        ut.probe(signal0)

        order = 190
        x = signal1
        noise_space = kernel_signalspace(x, order, mode='full', kernel='rbf',kpar=0.0001, use_ev=True)
        psd = subspace2psd(noise_space )
        plt.plot(psd); plt.show()

        order = 3
        x = signal0
        noise_space = kernel_signalspace(x, order, mode='full', kernel='rbf',kpar=0.0001, use_ev=True)
        freqs = subspace2freq(noise_space,order )
        print(freqs)
        
    References
    ----------------
    [1a] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page.
    [2a] M.H. Hayes. 
        Statistical Digital Signal Processing and Modeling, 
        John Wiley & Sons, 1996.
    [2b] https://www.mathworks.com/matlabcentral/fileexchange/
        2183-statistical-digital-signal-processing-and-modeling.
    
    '''     
    x = np.asarray(x)
    N = x.shape[0]       

    if(lags is None): lags = N//2

    R = operators.kernel_matrix(x,
                                mode   = mode,
                                kernel = kernel,
                                kpar   = kpar,
                                lags   = lags,
                                ret_base=False,
                                normalization=True)
    
       
    #TODO: in my practice eigen value always sorted
    # from the highest to the lowest, but probably sorting is required
    es,ev=np.linalg.eig(R)   
#     idxs = es.argsort()[::-1]
#     ev = ev[:,idxs]
    
    if use_ev:
        signal_space = np.matrix(ev[:,:order]/es[:order])
    else:
        signal_space = np.matrix(ev[:,:order])

    signal_space = np.eye(lags) - np.dot(signal_space, signal_space.H)
    
    return signal_space

# #------------------------------------
# def subspace2psd(subspace, n_psd = None ):
#     '''  
#     Subspace to pseudospetrum (psd) transformation,
#         uisng maximum entropy principle. 

#     Parameters
#     --------------
#     * subspace: 2d ndarray,
#         input subspace with 
#         dimentions = [lags,space_size].
#     * n_psd: int or None,
#         number of points in estimated psd, 
#         (subspace.shape[0]*2 by default).
    
#     Returns
#     -------------
#     * psd: 1d ndarray,
#         pseudospectrum of input subspace.
   
#     Notes
#     -----------
#     * If signal is real-valued, 
#         than order of the model shold be douled.
    
#     Example
#     -------------
   
#     References
#     ----------------
#     [1a] Stoica, Petre, and Randolph L. Moses. 
#         "Spectral analysis of signals." (2005).
#     [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
#         - Dr.Moses Spectral Analysis of Signals: Resource Page
#     [2a] M.H. Hayes. 
#         Statistical Digital Signal Processing and Modeling, 
#         John Wiley & Sons, 1996.
#     [2b] https://www.mathworks.com/matlabcentral/fileexchange/
#         2183-statistical-digital-signal-processing-and-modeling
    
#     '''
    
#     if n_psd is None: n_psd = 2*subspace.shape[0]

#     psd = 0

#     for i in range(subspace.shape[1]):
#         psd += np.abs( np.fft.ifft(subspace[:,i],n_psd, axis=0))
    
#     return 1/psd



# #------------------------------------
# def subspace2freq(subspace,order=None,fs=1):
#     '''  
#     Subspace to pseudospetrum (psd) transformation,
#         using alternative method. 

#     Parameters
#     --------------
#     * subspace: 2d ndarray,
#         input subspace with 
#         dimentions = [lags,space_size].
#     * order: int or None,
#         order of the model,
#         subspace.shape[1] - subspace.shape[0],
#         for default (if noise subspace is used).
#     fs: float,
#         sampling frequency.
    
#     Returns
#     -------------
#     * psd: 1d ndarray,
#         pseudospectrum of input subspace.
   
#     Notes
#     -----------


#     Example
#     -------------
   
#     References
#     ----------------
#     [1a] Stoica, Petre, and Randolph L. Moses. 
#         "Spectral analysis of signals." (2005).
#     [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
#         - Dr.Moses Spectral Analysis of Signals: Resource Page
#     [2a] M.H. Hayes. 
#         Statistical Digital Signal Processing and Modeling, 
#         John Wiley & Sons, 1996.
#     [2b] https://www.mathworks.com/matlabcentral/fileexchange/
#         2183-statistical-digital-signal-processing-and-modeling
        
#     '''
#     if fs is None: fs = 1
    
#     subspace = np.matrix(subspace)
    
#     if order is None:
#         order = subspace.shape[0]-subspace.shape[1]
        
#     P=subspace*subspace.H
#     N = subspace.shape[0]*2

#     Q=np.zeros(N,dtype=P.dtype)
#     for idx,val in enumerate(np.arange(N//2-1,-N//2,-1)):
#         Q[idx]=np.sum(np.diag(P,val))

#     roots = np.roots(Q)

#     roots = roots[np.abs(roots)<1]
#     roots = roots[np.imag(roots) != 0]
 
#     # extract roots numbers nearest to the circle with radius 1
#     unicircle_idxs = np.argsort( np.abs(np.abs(roots)-1) )
#     roots = roots[unicircle_idxs[:order]]

#     f= -fs*np.angle(roots)/(2.*np.pi)

#     return f[f>0]

