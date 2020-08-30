import numpy as np
import scipy

from ._kernel_periodogram import kernel_periodogram

_EPS_= 1e-5
#---------------------------------------------
def periodogram(x, window = None, x_range=None, n_psd = None):
    ''' 
    Periodogram-windowed spetrum estimation.
     
    Parameters 
    -----------------------
    * x: 1d ndarray.    
    * window: string or tuple(string, float)
        window type (square window if None).
    * x_range: [int,int] or int or None,
        if not none, determin first and last points
        of input signal to estimate (if only one value - 
        range will be taken as [0,x_range].   
    * n_psd: int or None, 
        Length of psceudo-spectrum 
                (Npsd = x.shape[0] if None).
    
    Returns
    -----------------------
    * periodogram: 1d ndarray.
    
    Notes
    ------------
    Scipy Window types:
        - `~scipy.signal.windows.boxcar`
        - `~scipy.signal.windows.triang`
        - `~scipy.signal.windows.blackman`
        - `~scipy.signal.windows.hamming`
        - `~scipy.signal.windows.hann`
        - `~scipy.signal.windows.bartlett`
        - `~scipy.signal.windows.flattop`
        - `~scipy.signal.windows.parzen`
        - `~scipy.signal.windows.bohman`
        - `~scipy.signal.windows.blackmanharris`
        - `~scipy.signal.windows.nuttall`
        - `~scipy.signal.windows.barthann`
        - `~scipy.signal.windows.kaiser` (needs beta)
        - `~scipy.signal.windows.gaussian` (needs standard deviation)
        - `~scipy.signal.windows.general_gaussian` (needs power, width)
        - `~scipy.signal.windows.slepian` (needs width)
        - `~scipy.signal.windows.dpss` (needs normalized half-bandwidth)
        - `~scipy.signal.windows.chebwin` (needs attenuation)
        - `~scipy.signal.windows.exponential` (needs decay scale)
        - `~scipy.signal.windows.tukey` (needs taper fraction)  
        
    References
    --------------------
    [1a] M.H. Hayes. Statistical Digital 
        Signal Processing and Modeling, John Wiley & Sons, 1996.
    [1b] https://www.mathworks.com/matlabcentral/fileexchange/2183
                -statistical-digital-signal-processing-and-modeling.
    [2a] P. Stoica, R.L. Moses, Spectral analysis of signals 
                        - New-York: Present-Hall, 2005.
    [2b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
            - Dr.Moses Spectral Analysis of Signals: Resource Page.
    [3]  S.L. Marple, Digital spectral analysis with applications 
                    – New-York: Present-Hall, 1986.        
    Example
    ---------------------- 
    
    See also
    ----------------------
    correlogram
    bartlett
    welch
    blackman_tukey
    daniell
    kernel_periodogram
    
    ''' 
    x = np.asarray(x)
    N = x.shape[0]
    
    if x_range == None: 
        x_range = np.asarray([0,N])
    else:
        x_range = np.asarray(x_range)    
        if x_range.size==1:
            x_range = np.append([0],[x_range])
    
    if(n_psd is None):n_psd = N
    
    x1 = x[x_range[0]:x_range[1]]
    if window is not None:
        w  = scipy.signal.get_window(window, x_range[1]-x_range[0])
        x1 = x1*np.conj(w)/np.linalg.norm(w)
        
    sp = np.fft.fft(x1, int(n_psd))
    sp[0] = sp[1]    
    return (sp*np.conj(sp)).real

#---------------------------------------------
def correlogram(x, n_psd = None):
    ''' 
    Correlogram-windowed spetrum estimation.
     
    Parameters 
    -----------------------
    * x: 1d ndarray of size N.    
    * n_psd:  Length of psceudo-spectrum 
                (Npsd = x.shape[0] if None).
    
    Returns
    -----------------------
    * periodogram estimation 1d ndarray.

    References
    --------------------
    [1a] P. Stoica, R.L. Moses, Spectral analysis of signals 
                        - New-York: Present-Hall, 2005.
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
            - Dr.Moses Spectral Analysis of Signals: Resource Page.
    
    Example
    ---------------------- 
    
    See also
    ----------------------
    periodogram
    bartlett
    welch
    blackman_tukey
    daniell
    kernel_periodogram
    ''' 
     
    return periodogram(x, window = None, x_range=None, n_psd = n_psd)

#---------------------------------------------
def bartlett(x, n_sections=1, n_psd = None):
    ''' 
    Periodogram-spetrum estimation based on the
        Bartlett's method.
     
    Parameters 
    -----------------------
    * x: 1d ndarray of size N.    
    * n_sections: number of sections to estimate.
    * n_psd:  Length of psceudo-spectrum 
                (Npsd = x.shape[0] if None).
    
    Returns
    -----------------------
    * periodogram estimation 1d ndarray.
        
    References
    --------------------
    [1a] M.H. Hayes. Statistical Digital 
        Signal Processing and Modeling, John Wiley & Sons, 1996.
    [1b] https://www.mathworks.com/matlabcentral/fileexchange/2183
                -statistical-digital-signal-processing-and-modeling.
    
    Example
    ---------------------- 
    
    See also
    ----------------------

    
    ''' 
    x = np.asarray(x)
    N = x.shape[0]
    L = N//n_sections
    if(n_psd is None): n_psd = N
    n_psd = int(n_psd) 
    px = 0
    n1 = 0
    
    for i in range(n_sections):
        px += periodogram(x, x_range=[n1,n1+L], window = None, n_psd = None)
        n1 += L    
    return px
#---------------------------------------------
def welch(x, window = None, window_length = None, step = None, n_psd = None):
    ''' 
    Welch modified Periodogram-windowed with overlay spetrum estimation.
     
    Parameters 
    -----------------------
    * x: 1d ndarray of size N.    
    * window: window type (square window if None).
    * window_length: number of section to take 
            (up to x.size-1, x.size-1 if None).
    * step: step of section moving (x.size if None).
    * n_psd:  Length of psceudo-spectrum 
                (Npsd = x.shape[0] if None).
    
    Returns
    -----------------------
    * periodogram estimation 1d ndarray.
    
    Notes
    ------------
    Scipy Window types:
        - `~scipy.signal.windows.boxcar`
        - `~scipy.signal.windows.triang`
        - `~scipy.signal.windows.blackman`
        - `~scipy.signal.windows.hamming`
        - `~scipy.signal.windows.hann`
        - `~scipy.signal.windows.bartlett`
        - `~scipy.signal.windows.flattop`
        - `~scipy.signal.windows.parzen`
        - `~scipy.signal.windows.bohman`
        - `~scipy.signal.windows.blackmanharris`
        - `~scipy.signal.windows.nuttall`
        - `~scipy.signal.windows.barthann`
        - `~scipy.signal.windows.kaiser` (needs beta)
        - `~scipy.signal.windows.gaussian` (needs standard deviation)
        - `~scipy.signal.windows.general_gaussian` (needs power, width)
        - `~scipy.signal.windows.slepian` (needs width)
        - `~scipy.signal.windows.dpss` (needs normalized half-bandwidth)
        - `~scipy.signal.windows.chebwin` (needs attenuation)
        - `~scipy.signal.windows.exponential` (needs decay scale)
        - `~scipy.signal.windows.tukey` (needs taper fraction)  
        
    References
    --------------------
    [1a] M.H. Hayes. Statistical Digital 
        Signal Processing and Modeling, John Wiley & Sons, 1996.
    [1b] https://www.mathworks.com/matlabcentral/fileexchange/2183
                -statistical-digital-signal-processing-and-modeling.
    [2a] P. Stoica, R.L. Moses, Spectral analysis of signals 
                        - New-York: Present-Hall, 2005.
    [2b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
            - Dr.Moses Spectral Analysis of Signals: Resource Page
    [3]  S.L. Marple, Digital spectral analysis with applications. 
                    – New-York: Present-Hall, 1986.      
    
    Example
    ---------------------- 
    
    See also
    ----------------------
    correlogram
    periodogram
    bartlett
    blackman_tukey
    daniell
    kernel_periodogram
    
    '''  
    
    x = np.asarray(x)
    N = x.shape[0]
    
    if(window_length is None or window_length>N-1): 
        window_length = N-1  
    if(step is None): step   = N        
    if(n_psd is None): n_psd = N
    

    S=int((N-window_length+step)/(step+_EPS_))
    
    px=0  

    for i in range(S+1):
        n1 = i*step
        n2 = min(int(n1 + window_length),N)
    
        px+= periodogram(x, window = window, 
                         x_range=[n1,n2], n_psd = n_psd )/S

    return px

#-------------------------------------------
def blackman_tukey(x,mode='full', window= None, lags = None, n_psd = None):
    '''
    Periodogram-windowed spetrum estimation based on the covariation matrix
        (Blackman-Tukey method).
     
    Parameters 
    -----------------------
    * x: 1d ndarray of size N.    
    * mode: covariance matrix mode,
        mode = {full,covar,traj,toeplitz}.
    * window: window type (square window if None).
    * lags: number of lags in kernel (x.shape[0]//2 if None).
    * n_psd:  Length of psceudo-spectrum 
                (Npsd = x.shape[0] if None).
    
    Returns
    -----------------------
    * periodogram estimation 1d ndarray.

    Notes
    ------------
    Scipy Window types:
        - `~scipy.signal.windows.boxcar`
        - `~scipy.signal.windows.triang`
        - `~scipy.signal.windows.blackman`
        - `~scipy.signal.windows.hamming`
        - `~scipy.signal.windows.hann`
        - `~scipy.signal.windows.bartlett`
        - `~scipy.signal.windows.flattop`
        - `~scipy.signal.windows.parzen`
        - `~scipy.signal.windows.bohman`
        - `~scipy.signal.windows.blackmanharris`
        - `~scipy.signal.windows.nuttall`
        - `~scipy.signal.windows.barthann`
        - `~scipy.signal.windows.kaiser` (needs beta)
        - `~scipy.signal.windows.gaussian` (needs standard deviation)
        - `~scipy.signal.windows.general_gaussian` (needs power, width)
        - `~scipy.signal.windows.slepian` (needs width)
        - `~scipy.signal.windows.dpss` (needs normalized half-bandwidth)
        - `~scipy.signal.windows.chebwin` (needs attenuation)
        - `~scipy.signal.windows.exponential` (needs decay scale)
        - `~scipy.signal.windows.tukey` (needs taper fraction)
    
            
    References
    --------------------
    [1a] M.H. Hayes. Statistical Digital 
        Signal Processing and Modeling, John Wiley & Sons, 1996.
    [1b] https://www.mathworks.com/matlabcentral/fileexchange/2183
                -statistical-digital-signal-processing-and-modeling.
    
    Example
    ---------------------- 
    
    See also
    ----------------------
    correlogram
    periodogram
    bartlett
    welch
    daniell
    kernel_periodogram
    
    '''
    return kernel_periodogram(x, mode=mode, kernel='linear', kpar=1, 
                            window= window, lags = lags, n_psd = n_psd)

#-------------------------------------------
def daniell(x,n_average=None,n_psd=None):
    ''' 
    Periodogram moving-average spetrum estimation based
        on the Daniell method.
     
    Parameters 
    -----------------------
    * x: 1d ndarray of size N.    
     * window: window type (square window if None).
    * n_average - number of points to average
        (at least 1).   
    * n_psd:  Length of psceudo-spectrum 
                (Npsd = x.shape[0] if None).
    
    Returns
    -----------------------
    * periodogram estimation 1d ndarray.

        
    References
    --------------------
    [1a] P. Stoica, R.L. Moses, Spectral analysis of signals 
                        - New-York: Present-Hall, 2005.
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
            - Dr.Moses Spectral Analysis of Signals: Resource Page.
        
    Example
    ---------------------- 
    
    See also
    ----------------------
    correlogram
    periodogram
    bartlett
    welch
    blackman_tukey
    kernel_periodogram
    
    ''' 
    
    
    x = np.asarray(x)
    N = x.shape[0]
    
    if n_average is None: n_average = 1
    else: 
        if n_average<1: raise ValueError('n_average<1')
    if n_psd is None: n_psd=N    
        
    psd = periodogram(x, window = None, 
                         x_range=None, n_psd = n_psd )

    
    psd = np.concatenate(( psd[n_psd-n_average:n_psd],psd,psd[:n_average] ))
    
    b = np.ones(n_average)
    a = [1]
    
    psd = scipy.signal.lfilter(b,a,psd)/n_average
    return psd[2*n_average:]
# #---------------------------------------------
# def welch(x, window=None, n_sections=1, ovelay = 0, n_psd = None):
#     ''' 
#     Welch modified Periodogram-windowed with overlay spetrum estimation.
     
#     Parameters 
#     -----------------------
#     * x: 1d ndarray of size N.    
#     * window: window type (square window if None).
#     * n_sections: number of section to take.
#     * overlay: percent of section overlay.
#     * n_psd:  Length of psceudo-spectrum 
#                 (Npsd = x.shape[0] if None).
    
#     Returns
#     -----------------------
#     * periodogram estimation 1d ndarray.
    
#     Notes
#     ------------
#     Scipy Window types:
#         - `~scipy.signal.windows.boxcar`
#         - `~scipy.signal.windows.triang`
#         - `~scipy.signal.windows.blackman`
#         - `~scipy.signal.windows.hamming`
#         - `~scipy.signal.windows.hann`
#         - `~scipy.signal.windows.bartlett`
#         - `~scipy.signal.windows.flattop`
#         - `~scipy.signal.windows.parzen`
#         - `~scipy.signal.windows.bohman`
#         - `~scipy.signal.windows.blackmanharris`
#         - `~scipy.signal.windows.nuttall`
#         - `~scipy.signal.windows.barthann`
#         - `~scipy.signal.windows.kaiser` (needs beta)
#         - `~scipy.signal.windows.gaussian` (needs standard deviation)
#         - `~scipy.signal.windows.general_gaussian` (needs power, width)
#         - `~scipy.signal.windows.slepian` (needs width)
#         - `~scipy.signal.windows.dpss` (needs normalized half-bandwidth)
#         - `~scipy.signal.windows.chebwin` (needs attenuation)
#         - `~scipy.signal.windows.exponential` (needs decay scale)
#         - `~scipy.signal.windows.tukey` (needs taper fraction)  
        
#     References
#     --------------------
#     [1a] M.H. Hayes. Statistical Digital 
#         Signal Processing and Modeling, John Wiley & Sons, 1996.
#     [1b] https://www.mathworks.com/matlabcentral/fileexchange/2183
#                 -statistical-digital-signal-processing-and-modeling
    
#     Example
#     ---------------------- 
    
#     See also
#     ----------------------

    
#     '''   
#     x = np.asarray(x)
#     N = x.shape[0]
#     n1 = 0    
#     n2 = N//n_sections
#     n0 = (n2*(1-ovelay))#np.round((1-ovelay)*n_sections)
    
    
#     nsect=1+int((N-n2)/(n0))


#     px=0    
#     for i in range(nsect+1):
#         print(n1,n2)
#         px+= periodogram(x, window = window, 
#                          x_range=[n1,n2], n_psd = n_psd )/nsect
#         n1 = int(n1 + n0)
#         n2 = min(int(n2 + n0),N)

#     return px

# #---------------------------------------------
# def cormatrix_by_Stoica(y,m = None, take_mean = True, unbias = True, ):
    
#     y = np.asarray(y)
#     N = y.shape[0]
    
# #     if(take_mean):
# #         y -=np.mean(y)
    
    
#     if(m is None):
#         m = N//2
    
#     R=np.zeros((m,m))
    
#     for i in np.arange(m, N):
#         R=R+np.outer(y[i:i-m:-1],np.conj(y[i:i-m:-1]))
        
# #     if(unbias):
# #         R /=(N-m - np.arange(m, N)+1)
# #     else:
#     R /=(N-m)
#     return R


# amplitudes = [2, 0.0,0.0]
# delays1    = [40,160,158]
# dev_freqs  = [0.01,0.3,0.0 ]

# SNR = 26
# x = make_beat_sig(amplitudes,delays1,dev_freqs, SNRdB = SNR)
# ut.probe(x)
# plt.plot(ut.afft(x)[:100]);plt.show()

# c = pmusic(x, N_of_components = 5,p_sart=3)

# cap = capon(x,order=3)

# print(rootmusic(x, N_of_components = 6,p_sart=1,unbias=True,FB=True))
# # print(esprit(x, N_of_components = 16,p_sart=0,unbias=False,FB=True))

# # print(pisarenko1freq(x))

# plt.plot(ut.afft(x)*np.max(np.abs(cap[:]))/np.max(ut.afft(x)),'--')
# # plt.plot(np.abs(c[:]),'k')
# plt.plot(np.abs(cap[:]),'k')

# plt.show()

# def capone(x, order, fs =1,Nlags=None, unbias=True,    FB=True, Nfft = None):
#     ''' from Stoica ch5'''
#     x = np.asarray(x)
#     N = x.shape[0]
    
#     if(not Nlags):
#         Nlags = N//2
    
#     R  = correlation.corr_matrix(x, order,  take_mean=True, unbias=unbias,    FB=FB) #cormatrix_by_Stoica(x, m = order, )
    
#     if(Nfft is None):
#         Nfft = N
        
        
#     Nfft = int(Nfft)    
   
#     R = np.matrix(R)
#     IR = np.linalg.inv(R)
#     phi = np.zeros(Nfft)
        
#     frange = np.arange(Nfft)    
#     pseudospetrum = np.zeros(Nfft, dtype='complex')

#     for i in np.arange(Nfft):
#         fi = fs*i/Nfft
#         a=[np.exp(- 2j*np.pi*fi*n/fs) for n in range(order) ]

#         a = np.matrix(a,dtype='complex').T
        
#         pseudospetrum[i] = (order+1)/np.real(a.H*IR*a)
        
#     return  pseudospetrum
