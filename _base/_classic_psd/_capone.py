import numpy as np
import scipy

from ... import operators

#---------------------------------------------
def capone(x, order, mode = 'full', 
           lags = None, n_psd = None):
    ''' 
    Capone filter for pseudo-spetrum estimation.
     
    Parameters 
    -------------
    * x: 1d ndarray 
        input signal of size N.    
    * order: int,
        number of components to extract.
    * mode: string,
        mode of covariation matrix,
        mode = {full, toeplitz, covar, mcoavr, prew, postw, traj}.    
    * lags: int or None,
        Number of lags in correlation function, 
        (x.shape[0]//2 by default).
    * n_psd: int or None,
        Length of psceudo-spectrum (Npsd = x.shape[0] if None).
    
    Returns
    -----------------------
    * psceudo-spectrum: 1d ndarray.
        
    Refernce
    ----------------------- 
    [1] Stoica, Petre, and Randolph L. Moses. 
        "Spectral analysis of signals." (2005).
    [2] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
        - Dr.Moses Spectral Analysis of Signals: Resource Page
    
    Example
    ---------------------- 
    
    See also
    ----------------------
    slepian
    operators.covariance_matrix for learn modes.
    
    ''' 
    x = np.asarray(x)
    N = x.shape[0]
    
    if lags==None:  lags=N//2
    if(n_psd is None): n_psd = N
        
    n_psd = int(n_psd)    
    
    R = operators.covariance_matrix(x,mode=mode, lags=lags) 
    R = np.matrix(R[:order,:order])

    R = np.matrix(R)
    IR = np.linalg.inv(R)
    phi = np.zeros(n_psd)
        
    frange = np.arange(n_psd)    
    pseudospetrum = np.zeros(n_psd, dtype=np.complex)
    
    n = np.arange(order)
    for i in np.arange(n_psd):
        a = np.exp(- 2j*np.pi*i*n/n_psd) 
        a = np.matrix(a,dtype=np.complex).T
  
        pseudospetrum[i] =((order+1)/(a.H*IR*a))
        
    return  np.abs(pseudospetrum)


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
